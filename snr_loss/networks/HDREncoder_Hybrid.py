import torch
import torch.nn as nn
import torch.nn.functional as F
from .SwinTransformer import SwinTransformerV2Encoder
from .HDREncoder import UEncoder4Recurrent, FusionLayer_Unet


class AdaptiveFusion(nn.Module):
    """自适应特征融合模块"""

    def __init__(self, local_channels, global_channels):
        super().__init__()
        # 局部特征路径
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_channels, local_channels, 3, padding=1),
            nn.InstanceNorm2d(local_channels),
            nn.ReLU(inplace=True),
        )

        # 全局特征路径
        self.global_conv = nn.Sequential(
            nn.Conv2d(global_channels, global_channels, 3, padding=1),
            nn.InstanceNorm2d(global_channels),
            nn.ReLU(inplace=True),
        )

        # 自适应权重
        self.attention = nn.Sequential(
            nn.Conv2d(local_channels + global_channels, local_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_channels // 4, 2, 1),
            nn.Softmax(dim=1),
        )

        # 输出融合
        self.output_conv = nn.Conv2d(
            local_channels + global_channels, local_channels, 1
        )

    def forward(self, local_feat, global_feat):
        """
        Args:
            local_feat: [B, C1, H, W] - ConvLSTM特征（细节丰富）
            global_feat: [B, C2, H, W] - Swin特征（全局语义）
        Returns:
            fused: [B, C1, H, W] - 融合特征
        """
        # 处理两个分支
        local_feat = self.local_conv(local_feat)
        global_feat = self.global_conv(global_feat)

        # 计算自适应权重
        concat = torch.cat([local_feat, global_feat], dim=1)
        weights = self.attention(concat)  # [B, 2, H, W]

        # 加权融合
        fused = weights[:, 0:1] * local_feat + weights[:, 1:2] * global_feat

        # 残差连接
        output = self.output_conv(concat) + fused

        return output


class HDRev_Encoder_Hybrid(nn.Module):
    """混合架构：ConvLSTM（细节）+ Swin Transformer（全局）

    架构设计：
    1. 低层：ConvLSTM保留空间细节，使用预训练权重
    2. 高层：Swin建模全局语义，使用ImageNet预训练
    3. 自适应融合：根据特征重要性动态融合

    优势：
    - 保持ConvLSTM的细节保留能力
    - 引入Swin的全局建模能力
    - 两全其美，效果更好
    """

    def __init__(
        self,
        num_bins=5,
        pretrained_convlstm_I_path="pretrained/HDRev/Encoder_I.pth",
        pretrained_convlstm_E_path="pretrained/HDRev/Encoder_E.pth",
        pretrained_swin_I_path="pretrained/SwinTransformer/swinv2_base_patch4_window12_192_22k.pth",
        pretrained_swin_E_path="pretrained/SwinTransformer/swinv2_base_patch4_window12_192_22k.pth",
        pretrained_merge_path="pretrained/HDRev/Merge.pth",
        img_size=192,
        target_channels=[64, 128, 256, 512],
    ):
        super().__init__()

        self.img_size = img_size
        self.target_channels = target_channels

        # ==================== 局部细节分支：ConvLSTM ====================
        print("Loading ConvLSTM encoders for local detail preservation...")
        self.local_encoder_I = UEncoder4Recurrent(in_channels=3)
        if pretrained_convlstm_I_path and pretrained_convlstm_I_path != "":
            try:
                self.local_encoder_I.load_state_dict(
                    torch.load(
                        pretrained_convlstm_I_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                print(
                    f"✓ Loaded ConvLSTM Image encoder from {pretrained_convlstm_I_path}"
                )
            except Exception as e:
                print(f"⚠ Warning: Could not load ConvLSTM Image encoder: {e}")

        self.local_encoder_E = UEncoder4Recurrent(in_channels=num_bins)
        if pretrained_convlstm_E_path and pretrained_convlstm_E_path != "":
            try:
                self.local_encoder_E.load_state_dict(
                    torch.load(
                        pretrained_convlstm_E_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                print(
                    f"✓ Loaded ConvLSTM Event encoder from {pretrained_convlstm_E_path}"
                )
            except Exception as e:
                print(f"⚠ Warning: Could not load ConvLSTM Event encoder: {e}")

        # ==================== 全局语义分支：Swin Transformer ====================
        print("Loading Swin Transformer for global context modeling...")
        self.global_encoder_I = SwinTransformerV2Encoder(
            img_size=img_size,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.2,
            target_channels=target_channels,
            pretrained_path=pretrained_swin_I_path if pretrained_swin_I_path else None,
        )

        self.event_preprocess = nn.Sequential(
            nn.Conv2d(num_bins, 3, kernel_size=1, bias=False),
            nn.InstanceNorm2d(3),
        )

        self.global_encoder_E = SwinTransformerV2Encoder(
            img_size=img_size,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.2,
            target_channels=target_channels,
            pretrained_path=pretrained_swin_E_path if pretrained_swin_E_path else None,
        )

        # ==================== 自适应融合模块 ====================
        print("Initializing adaptive fusion modules...")
        self.fusion_modules = nn.ModuleList(
            [
                AdaptiveFusion(
                    local_channels=target_channels[i],
                    global_channels=target_channels[i],
                )
                for i in range(4)
            ]
        )

        # 最终融合层（使用原始的FusionLayer）
        self.final_fusion = FusionLayer_Unet(n_features=4)
        if pretrained_merge_path and pretrained_merge_path != "":
            try:
                self.final_fusion.load_state_dict(
                    torch.load(
                        pretrained_merge_path, map_location="cpu", weights_only=True
                    )
                )
                print(f"✓ Loaded Fusion layer from {pretrained_merge_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load Fusion layer: {e}")

        self.statei_local, self.statee_local = None, None
        self.dtype = torch.float32

    def forward(self, ldr, evs, return_img=False):
        """混合编码器前向传播

        Args:
            ldr: [B, 3, H, W] - LDR图像
            evs: [B, num_bins, H, W] - 事件数据
            return_img: 是否返回重建图像

        Returns:
            feat_final: [B, 512, H/8, W/8] - 最终特征
            feat_list: List of 4 multi-scale features
        """
        # Flip LDR (与原始保持一致)
        ldr = torch.flip(ldr, dims=[1])

        # 保存原始尺寸
        B, C, H_orig, W_orig = ldr.shape
        original_size = (H_orig, W_orig)

        # ==================== 局部分支：ConvLSTM ====================
        # ConvLSTM直接处理原始分辨率
        local_feat_i, self.statei_local = self.local_encoder_I(ldr, self.statei_local)
        local_feat_e, self.statee_local = self.local_encoder_E(evs, self.statee_local)

        # ConvLSTM特征已经是正确尺寸：[H, H/2, H/4, H/8]
        # local_feat_i: List of [B, C, H, W], [B, C, H/2, W/2], [B, C, H/4, W/4], [B, C, H/8, W/8]

        # ==================== 全局分支：Swin Transformer ====================
        # Swin需要resize到预训练尺寸
        if H_orig != self.img_size or W_orig != self.img_size:
            ldr_resized = F.interpolate(
                ldr,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            evs_resized = F.interpolate(
                evs,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            ldr_resized = ldr
            evs_resized = evs

        # 提取全局特征
        evs_preprocessed = self.event_preprocess(evs_resized)
        global_feat_i = self.global_encoder_I(ldr_resized)  # 48x48, 24x24, 12x12, 6x6
        global_feat_e = self.global_encoder_E(evs_preprocessed)

        # 上采样全局特征到原始分辨率
        target_sizes = [
            original_size,
            (original_size[0] // 2, original_size[1] // 2),
            (original_size[0] // 4, original_size[1] // 4),
            (original_size[0] // 8, original_size[1] // 8),
        ]

        global_feat_i_upsampled = []
        global_feat_e_upsampled = []

        for i, (feat_i, feat_e, target_size) in enumerate(
            zip(global_feat_i, global_feat_e, target_sizes)
        ):
            feat_i_up = F.interpolate(
                feat_i, size=target_size, mode="bilinear", align_corners=False
            )
            feat_e_up = F.interpolate(
                feat_e, size=target_size, mode="bilinear", align_corners=False
            )
            global_feat_i_upsampled.append(feat_i_up)
            global_feat_e_upsampled.append(feat_e_up)

        # ==================== 自适应融合 ====================
        # 融合图像特征
        fused_feat_i = []
        for i, (local_i, global_i) in enumerate(
            zip(local_feat_i, global_feat_i_upsampled)
        ):
            # 确保尺寸匹配
            if local_i.shape[2:] != global_i.shape[2:]:
                global_i = F.interpolate(
                    global_i,
                    size=local_i.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            fused = self.fusion_modules[i](local_i, global_i)
            fused_feat_i.append(fused)

        # 融合事件特征
        fused_feat_e = []
        for i, (local_e, global_e) in enumerate(
            zip(local_feat_e, global_feat_e_upsampled)
        ):
            if local_e.shape[2:] != global_e.shape[2:]:
                global_e = F.interpolate(
                    global_e,
                    size=local_e.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            fused = self.fusion_modules[i](local_e, global_e)
            fused_feat_e.append(fused)

        # ==================== 最终融合 ====================
        # 使用原始的FusionLayer融合图像和事件特征
        feat_final = self.final_fusion(fused_feat_e, fused_feat_i)

        # 重置ConvLSTM状态
        self.statei_local, self.statee_local = None, None

        # ==================== 保护处理 ====================
        feat_safe = []
        for f in feat_final:
            # 1. 替换 nan/inf 为 0
            f = torch.where(torch.isnan(f) | torch.isinf(f), torch.zeros_like(f), f)

            # 2. 限制绝对值范围到 [-50, 50]
            f = torch.clamp(f, -50, 50)

            # 3. 使用 tanh 归一化到 [-1, 1]
            f = torch.tanh(f / 10.0)

            feat_safe.append(f)

        return feat_safe[-1], feat_safe
