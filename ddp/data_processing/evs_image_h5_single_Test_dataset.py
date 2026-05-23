import os
import glob
import h5py
import torch
import numpy as np
import torch.nn.functional as tF
from .base_dataset import BaseDataset
from utils import whiteBalance, paired_random_crop, rgb_to_grayscale_numpy

eps = 1e-8


class evsImageH5SingleTestDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument(
                "--train_size",
                type=int,
                default=16,
                help="the size n of training windows, [n * n]",
            )
        return parser

    def __init__(self, opt):
        """
        支持：
        - opt.dataroot = 单个 .h5/.hdf5 文件
        - opt.dataroot = 一个文件夹，里面有多个 .h5/.hdf5
        读取方式：每个 h5 的“顶层 key”（如 '0002'）视为一个样本。
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.num_bins = opt.num_bins
        self.event_representation = opt.event_representation
        self.event_norm = opt.event_norm

        if self.isTrain:
            self.patch_size = opt.patch_size
            self.under_over_ratio = opt.under_over_ratio

        self.dataroot = opt.dataroot

        # 输出尺寸
        if self.isTrain:
            self.height, self.width = (
                (self.patch_size, self.patch_size)
                if isinstance(self.patch_size, int)
                else self.patch_size
            )
        else:
            self.height, self.width = opt.patch_size

        # 1) 收集所有 h5 文件
        self.h5_paths = self._collect_h5_paths(self.dataroot)
        if len(self.h5_paths) == 0:
            raise FileNotFoundError(f"No .h5/.hdf5 found under: {self.dataroot}")

        # 2) 展平所有 (h5_path, top_key)
        self.samples = []
        for h5_path in self.h5_paths:
            with h5py.File(h5_path, "r") as f:
                keys = list(f.keys())
            for k in keys:
                self.samples.append((h5_path, k))

        # 3) 限制 max_dataset_size
        if getattr(opt, "max_dataset_size", -1) != -1:
            self.samples = self.samples[: opt.max_dataset_size]

        self.dataset_size = len(self.samples)

    def _collect_h5_paths(self, dataroot):
        # dataroot 是文件
        if os.path.isfile(dataroot):
            return [dataroot]

        # dataroot 是文件夹
        if os.path.isdir(dataroot):
            h5s = glob.glob(os.path.join(dataroot, "*.h5"))
            hdf5s = glob.glob(os.path.join(dataroot, "*.hdf5"))
            return sorted(list(set(h5s + hdf5s)))

        raise FileNotFoundError(f"dataroot not found: {dataroot}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        h5_path, top_key = self.samples[index]

        with h5py.File(h5_path, "r") as f:
            group = f[top_key]

            img = np.array(group["gt"])
            ldr = np.array(group["ldr"])

            # 事件键：你现在用 evs_tuple；如果你的某些 h5 用的是 evs，就做个兼容
            if "evs_tuple" in group:
                event_data = np.array(group["evs_tuple"])
            elif "evs" in group:
                event_data = np.array(group["evs"])
            else:
                raise KeyError(
                    f"[{os.path.basename(h5_path)} / {top_key}] missing 'evs_tuple' or 'evs'"
                )

        # --- 后处理（保持你原逻辑） ---
        img = whiteBalance(img)
        img = np.transpose(img, (2, 0, 1))
        gray_img = rgb_to_grayscale_numpy(img)
        proc_gray = gray_img / (np.mean(gray_img) + eps) * 0.1
        img = img / (gray_img + eps) * proc_gray

        ldr = np.clip(ldr, 0, 1)
        ldr = np.transpose(ldr, (2, 0, 1))

        ev_height, ev_width = ldr.shape[1:]
        t, x, y, p = (
            event_data[:, 0],
            event_data[:, 1],
            event_data[:, 2],
            event_data[:, 3],
        )
        p = p * 2 - 1

        event_representation = self.__events_to_voxel_grid(
            x, y, p, t, self.num_bins, ev_width, ev_height
        )

        if self.event_norm and np.any(event_representation != 0):
            non_zero = event_representation[event_representation != 0]
            mean, stddev = non_zero.mean(), non_zero.std()
            if stddev > eps:
                event_representation[event_representation != 0] = (
                    non_zero - mean
                ) / stddev

        event_representation = torch.from_numpy(event_representation.copy())
        ldr = torch.from_numpy(ldr.copy()).float()
        img = torch.from_numpy(img.copy()).float()

        if self.isTrain:
            img, [event_representation, ldr] = paired_random_crop(
                img.permute(1, 2, 0),
                [event_representation.permute(1, 2, 0), ldr.permute(1, 2, 0)],
                self.height,
                None,
            )
            img = img.permute(2, 0, 1)
            event_representation = event_representation.permute(2, 0, 1)
            ldr = ldr.permute(2, 0, 1)
        else:
            img = tF.interpolate(
                img.unsqueeze(0),
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            event_representation = tF.interpolate(
                event_representation.unsqueeze(0),
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            ldr = tF.interpolate(
                ldr.unsqueeze(0),
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[0]

        # 避免不同 h5 里 top_key 重名：save_path 带上 h5 文件名
        h5_stem = os.path.splitext(os.path.basename(h5_path))[0]
        save_path = f"{h5_stem}/{top_key}"

        return {
            "save_path": save_path,
            "pixel_events": event_representation,
            "pixel_images": ldr,
            "gts": img,
            "text": "",
            "negative_text": "",
        }

    def __events_to_voxel_grid(self, x, y, p, t, num_bins, width, height):
        assert num_bins > 0 and width > 0 and height > 0
        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
        if len(x) == 0:
            return np.reshape(voxel_grid, (num_bins, height, width))

        last_stamp = t[-1]
        first_stamp = t[0]
        deltaT = last_stamp - first_stamp
        if deltaT == 0:
            deltaT = 1.0

        mask = (x < width) & (x >= 0) & (y < height) & (y >= 0)
        t = (num_bins - 1) * (t - first_stamp) / deltaT

        ts = t[mask]
        xs = x.astype(int)[mask]
        ys = y.astype(int)[mask]
        pols = p[mask]
        pols[pols == 0] = -1

        tis = ts.astype(int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid = tis < num_bins
        np.add.at(
            voxel_grid,
            xs[valid] + ys[valid] * width + tis[valid] * width * height,
            vals_left[valid],
        )

        valid = (tis + 1) < num_bins
        np.add.at(
            voxel_grid,
            xs[valid] + ys[valid] * width + (tis[valid] + 1) * width * height,
            vals_right[valid],
        )

        return np.reshape(voxel_grid, (num_bins, height, width))
