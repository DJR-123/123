import datetime
import argparse
import wandb
import math
import os

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler

from data_processing import create_dataset
from utils import save_image, VGGLoss, GANLoss
from networks import HDRev_Encoder, OursAutoencoderKL

eps = 1e-8


def get_condition_input(config, batch):
    if config.control_type == "evs":
        ret = batch["pixel_events"]
    elif config.control_type == "ldr":
        ret = batch["pixel_images"]
    elif config.control_type == "evs+ldr":
        ret = torch.cat([batch["pixel_events"], batch["pixel_images"]], dim=1)
    else:
        raise NotImplementedError(f"Not implemented control type")
    return ret


def main(name, config, use_wandb=False, debug=False, pretrained=("", "")):
    pretrained_unet, pretrained_upsample = pretrained
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    device = torch.device("cuda", local_rank)

    # ============ 使用混合精度训练 ============
    scaler = GradScaler(device="cuda")
    use_amp = False  # 开启自动混合精度
    # ========================================

    # create checkpoints and folders
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = f"debug" if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)

    # create scheduler and models
    cond_encoder = HDRev_Encoder(num_bins=config.train_dataset.num_bins)
    upsampler = OursAutoencoderKL.from_pretrained(
        config.pretrained_model_path,
        cache_dir="pretrained",
        subfolder="vae",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_path,
        cache_dir="pretrained",
        subfolder="vae",
        low_cpu_mem_usage=False,
        device_map=None,
    )

    for name, param in upsampler.named_parameters():
        if "fusion" in name:
            # param.requires_grad = True
            if "encode_enc_3.conv_out" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.constant_(param, 1e-6)

    if pretrained_unet != "":
        if not os.path.exists(pretrained_unet):
            raise ValueError(f"pretrained file {pretrained_unet} not exists.")
        print(f"load state dict from {pretrained_unet}")

        state_dict_cond = torch.load(pretrained_unet, map_location="cpu")[
            "state_dict_cond"
        ]
        m, u = cond_encoder.load_state_dict(state_dict_cond)
        print(m, u)
        print(
            f"cond_encoder:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}"
        )
    else:
        print("Not specify base model")
        exit()

    if pretrained_upsample != "":
        if not os.path.exists(pretrained_upsample):
            raise ValueError(f"pretrained file {pretrained_upsample} not exists.")
        print(f"load state dict from {pretrained_upsample}")

        state_dict_upsample = torch.load(pretrained_upsample, map_location="cpu")[
            "state_dict"
        ]
        m, u = upsampler.load_state_dict(state_dict_upsample)
        print(m, u)
        print(
            f"upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}"
        )

    # process trainable and frozen params
    cond_encoder.requires_grad_(False)
    upsampler.requires_grad_(False)
    for name, param in upsampler.named_parameters():
        if "fusion" in name:
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, upsampler.parameters()))
    # crate optimizer
    gan_loss = GANLoss().to(device)
    gan_loss.requires_grad_(True)
    trainable_params += list(filter(lambda p: p.requires_grad, gan_loss.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate * world_size,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # move to GPU
    cond_encoder.to(device)
    upsampler.to(device)
    vae.to(device)

    # Wrap models with DDP
    if world_size > 1:
        upsampler = DDP(
            upsampler,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        gan_loss = DDP(gan_loss, device_ids=[local_rank], output_device=local_rank)

    # create dataset and dataloader
    dataset = create_dataset(config.train_dataset)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            drop_last=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )

    max_train_steps = config.max_train_steps // world_size
    checkpointing_steps = config.checkpointing_steps
    gradient_accumulation_steps = config.gradient_accumulation_steps
    # diffusion iterations and learning rates

    lr_scheduler = get_scheduler(
        config.lr_sheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = config.batch_size * world_size * gradient_accumulation_steps

    if is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {config.batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")

    if (not debug) and use_wandb and is_main_process:
        wandb.init(project="HDR-diffu", name=folder_name, config=dict(config))

    if is_main_process:
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(os.path.join(out_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "checkpoints"), exist_ok=True)
        OmegaConf.save(config, os.path.join(out_folder, "config.yaml"))
        # save the code of training and the network
        code_dir = os.path.join(out_folder, "code")
        os.makedirs(os.path.join(out_folder, "code"), exist_ok=True)
        train_file = os.path.join("train_upsampler.py")
        os.system(f"cp {train_file} {code_dir}")
        os.system(f"cp -r networks {code_dir}")

    if world_size > 1:
        dist.barrier()

    global_step = 0
    first_epoch = 0

    if is_main_process:
        progress_bar = tqdm(range(global_step, max_train_steps))
        progress_bar.set_description("Steps")

    vgg_loss = VGGLoss().to(device)

    for epoch in range(first_epoch, num_train_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            # ============ 使用混合精度前向传播 ============
            with autocast("cuda", enabled=use_amp):
                # training
                new_gt = batch["gts"].to(device)
                latents = batch["latents"].to(device)

                condition_images, condition_list = cond_encoder(
                    batch["pixel_images"].to(device),
                    batch["pixel_events"].to(device),
                    return_img=False,
                )

                upsampler_model = upsampler.module if world_size > 1 else upsampler
                out_img = upsampler_model.decode(
                    latents / upsampler_model.config.scaling_factor, condition_list
                ).sample

                upsample_loss = (
                    vgg_loss(out_img, new_gt).mean() * 0.0001
                ) + F.mse_loss(out_img.float(), new_gt.float(), reduction="mean") * 0.01
            # ==========================================

            optimizer.zero_grad()

            # ============ 使用 scaler 反向传播 ============
            scaler.scale(upsample_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ==========================================

            lr_scheduler.step()
            global_step += 1
            if is_main_process:
                progress_bar.update(1)

            # logging
            if (not debug) and use_wandb and is_main_process:
                wandb.log(
                    {"upsampler_loss": upsample_loss.item(), "gan_loss": 0},
                    step=global_step,
                )

            # saving
            if (
                global_step % checkpointing_steps == 0 or step == len(dataloader) - 1
            ) and is_main_process:
                save_path = os.path.join(out_folder, "checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict_upsample": upsampler_model.state_dict(),
                }
                if step == len(dataloader) - 1:
                    if epoch % 10 == 0:
                        torch.save(
                            state_dict,
                            os.path.join(save_path, f"epoch-{epoch + 1}.ckpt"),
                        )
                else:
                    torch.save(state_dict, os.path.join(save_path, "latest.ckpt"))
                print(f"saving model to {save_path} with global step {global_step}")

            # validation
            if global_step % config.validation_steps == 0 and is_main_process:
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.global_seed)

                img_pred = vae.decode(latents / vae.config.scaling_factor).sample
                visuals = {}
                visuals[f"{global_step}_results_tm"] = (out_img + 1) / 2
                visuals[f"{global_step}_results_sample"] = (img_pred + 1) / 2
                visuals[f"{global_step}_results_diff"] = ((out_img - new_gt) + 2) / 4
                visuals[f"{global_step}_events"] = batch["pixel_events"]
                visuals[f"{global_step}_images"] = batch["pixel_images"]
                visuals[f"{global_step}_gt"] = (new_gt + 1) / 2
                save_path = os.path.join(out_folder, "images")
                save_image(visuals, save_path)
            logs = {
                "step_loss": upsample_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if is_main_process:
                progress_bar.set_postfix(**logs)

            if global_step > max_train_steps:
                break

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained_unet", type=str, default="")
    parser.add_argument("--pretrained_upsample", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    name = Path(args.config).stem + "_" + args.name

    config = OmegaConf.load(args.config)

    main(
        name=name,
        use_wandb=args.wandb,
        config=config,
        debug=args.debug,
        pretrained=(args.pretrained_unet, args.pretrained_upsample),
    )
