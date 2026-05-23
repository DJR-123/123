import torch

from networks import HDRev_Encoder_Hybrid


def create_encoder_from_config(config, dataset_config_key="train_dataset"):
    """
    根据配置创建encoder实例

    Args:
        config: OmegaConf配置对象
        dataset_config_key: 数据集配置键名 ('train_dataset' 或 'test_dataset')

    Returns:
        cond_encoder: 编码器实例
    """
    encoder_type = config.get("encoder_type", "hybrid")
    encoder_config = config.get("encoder_config", {})
    dataset_config = config.get(dataset_config_key, {})
    num_bins = dataset_config.get("num_bins", 5)

    if encoder_type == "hybrid":
        cond_encoder = HDRev_Encoder_Hybrid(
            num_bins=encoder_config.get("num_bins", num_bins),
            img_size=encoder_config.get("img_size", 192),
            target_channels=encoder_config.get("target_channels", [64, 128, 256, 512]),
            pretrained_convlstm_I_path=encoder_config.get(
                "pretrained_convlstm_I_path", "pretrained/HDRev/Encoder_I.pth"
            ),
            pretrained_convlstm_E_path=encoder_config.get(
                "pretrained_convlstm_E_path", "pretrained/HDRev/Encoder_E.pth"
            ),
            pretrained_swin_I_path=encoder_config.get(
                "pretrained_swin_I_path",
                "pretrained/SwinTransformer/swinv2_base_patch4_window12_192_22k.pth",
            ),
            pretrained_swin_E_path=encoder_config.get(
                "pretrained_swin_E_path",
                "pretrained/SwinTransformer/swinv2_base_patch4_window12_192_22k.pth",
            ),
            pretrained_merge_path=encoder_config.get(
                "pretrained_merge_path", "pretrained/HDRev/Merge.pth"
            ),
        )
        print("Using Hybrid encoder (ConvLSTM + Swin)")
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. Only 'hybrid' is supported."
        )

    return cond_encoder


def get_condition_input(config, batch):
    """
    根据配置获取条件输入

    Args:
        config: OmegaConf配置对象
        batch: 数据batch字典

    Returns:
        条件输入tensor
    """
    control_type = config.get("control_type", "evs+ldr")

    if control_type == "evs":
        return batch["pixel_events"]
    elif control_type == "ldr":
        return batch["pixel_images"]
    elif control_type == "evs+ldr":
        return torch.cat([batch["pixel_events"], batch["pixel_images"]], dim=1)
    else:
        raise NotImplementedError(f"Control type '{control_type}' not implemented")
