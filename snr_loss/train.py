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

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler

from data_processing import create_dataset
from utils import save_image, tonemap
from networks import (
    OursControlNetModel,
    HDRev_Encoder_Hybrid,
)
from pipeline import create_pipeline

eps = 1e-8


def compute_snr_weight(timesteps, noise_scheduler, gamma=5.0, weighting_type="min_snr"):
    """计算SNR加权系数
    
    Args:
        timesteps: 当前时间步 [batch_size]
        noise_scheduler: 噪声调度器
        gamma: 加权强度参数
        weighting_type: 加权类型
            - "min_snr": Min-SNR weighting (Stable Diffusion使用)
            - "inverse": 逆SNR加权，偏向高噪声
            - "sqrt": 平方根SNR加权
    
    Returns:
        weight: 加权系数 [batch_size]
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    if weighting_type == "min_snr":
        # Min-SNR weighting: 限制SNR范围并取平方根
        # 防止在极高SNR（极低噪声）时权重过大
        snr_clipped = torch.clamp(snr, min=1.0, max=gamma)
        weight = torch.sqrt(snr_clipped)
    elif weighting_type == "inverse":
        # 逆SNR加权：偏向高噪声时刻
        weight = 1.0 / (snr + 1e-8)
        weight = torch.clamp(weight, min=0.1, max=10.0)
    elif weighting_type == "sqrt":
        # 简单平方根SNR
        weight = torch.sqrt(snr + 1e-8)
    else:
        raise ValueError(f"Unknown weighting_type: {weighting_type}")
    
    return weight


def safe_condition_list(cond_list):
    """
    确保 condition_list 不包含 nan/inf，并限制在安全范围内
    这是防止 NaN 的关键保护函数
    """
    safe_list = []
    for c in cond_list:
        # 1. 替换 nan/inf 为 0
        c = torch.where(torch.isnan(c) | torch.isinf(c), torch.zeros_like(c), c)

        # 2. 硬限制范围到 [-5, 5]
        c = torch.clamp(c, -5, 5)

        # 3. 最后检查（双保险）
        if torch.isnan(c).any() or torch.isinf(c).any():
            print(
                "⚠️ Warning: condition still has nan/inf after protection, using zeros"
            )
            c = torch.zeros_like(c)

        safe_list.append(c)
    return safe_list


def main(name, config, use_wandb=False, debug=False, pretrained=""):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    device = torch.device("cuda", local_rank)

    # ============ 混合精度训练设置 ============
    scaler = GradScaler(device="cuda")
    use_amp = False  # 暂未启用自动混合精度，可根据需要开启
    # ========================================

    # create checkpoints and folders
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = "debug" if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)

    # create scheduler and models
    noise_scheduler = DDIMScheduler(
        **OmegaConf.to_container(config.noise_scheduler_kwargs)
    )

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_path,
        cache_dir="pretrained",
        subfolder="vae",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_path, cache_dir="pretrained", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_path, cache_dir="pretrained", subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_path,
        cache_dir="pretrained",
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    controlnet = OursControlNetModel.from_pretrained(
        config.pretrained_controlnet_model_path,
        cache_dir="pretrained",
        cross_attention_dim=(
            1024
            if config.noise_scheduler_kwargs.prediction_type == "v_prediction"
            else 768
        ),
        conditioning_channels=config.conditioning_channels,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    for name, module in controlnet.named_modules():
        if "controlnet_cond_embedding" in name:
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.00001)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    encoder_type = config.get("encoder_type", "hybrid")
    encoder_config = config.get("encoder_config", {})
    cond_encoder = HDRev_Encoder_Hybrid(
        num_bins=encoder_config.get("num_bins", config.train_dataset.num_bins),
        img_size=encoder_config.get("img_size", 192),
        target_channels=encoder_config.get("target_channels", [64, 128, 256, 512]),
        pretrained_convlstm_I_path=encoder_config.get(
            "pretrained_convlstm_I_path",
            "pretrained/HDRev/Encoder_I.pth",
        ),
        pretrained_convlstm_E_path=encoder_config.get(
            "pretrained_convlstm_E_path",
            "pretrained/HDRev/Encoder_E.pth",
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
            "pretrained_merge_path",
            "pretrained/HDRev/Merge.pth",
        ),
    )
    print("Using Hybrid encoder (ConvLSTM + Swin)")

    if pretrained != "":
        if not os.path.exists(pretrained):
            raise ValueError(f"pretrained file {pretrained} not exists.")
        print(f"load state dict from {pretrained}")

        state_dict_controlnet = torch.load(pretrained, map_location="cpu")[
            "state_dict_controlnet"
        ]
        m, u = controlnet.load_state_dict(state_dict_controlnet)
        print(m, u)
        print(
            f"controlnet:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}"
        )

        state_dict_cond = torch.load(pretrained, map_location="cpu")["state_dict_cond"]
        m, u = cond_encoder.load_state_dict(state_dict_cond)
        print(m, u)
        print(
            f"cond_encoder:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}"
        )

    # process trainable and frozen params
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    cond_encoder.requires_grad_(False)

    if config.train_controlnet:
        for name, param in controlnet.named_parameters():
            for module_name in config.controlnet_trainable_modules:
                if module_name in name:
                    param.requires_grad = True

    if config.get("train_encoder", False):
        encoder_trainable_modules = config.get("encoder_trainable_modules", [""])
        for name, param in cond_encoder.named_parameters():
            for module_name in encoder_trainable_modules:
                if module_name in name:
                    param.requires_grad = True

    trainable_params = []
    # crate optimizer
    trainable_params += list(
        filter(lambda p: p.requires_grad, controlnet.parameters())
    ) + list(filter(lambda p: p.requires_grad, cond_encoder.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate * world_size,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # enable gradient checkpointing
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
        # ControlNet uses custom gradient handling
        # ControlNet gradient is handled internally, no need to call enable_gradient_checkpointing()

    # move to GPU
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    controlnet.to(device)
    cond_encoder.to(device)

    # Wrap models with DDP
    if world_size > 1:
        controlnet = DDP(
            controlnet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        cond_encoder = DDP(
            cond_encoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

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

    lr_scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    kwargs = {
        "unet": unet,
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "controlnet": controlnet.module if world_size > 1 else controlnet,
        "cond_encoder": cond_encoder.module if world_size > 1 else cond_encoder,
        "scheduler": noise_scheduler,
        "isVal": True,
        "upsampler": vae,
        "upsampler_w": None,
    }
    validation_pipeline = create_pipeline(config.validation_pipeline, kwargs).to(device)
    validation_pipeline.enable_vae_slicing()

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
        print(f"  Mixed Precision Training: {use_amp}")

    if (not debug) and use_wandb and is_main_process:
        wandb.init(project="HDR-diffu", name=folder_name, config=dict(config))

    if is_main_process:
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(os.path.join(out_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "checkpoints"), exist_ok=True)
        OmegaConf.save(config, os.path.join(out_folder, "config.yaml"))

        # save the code of test and pipeline
        code_dir = os.path.join(out_folder, "code")
        os.makedirs(os.path.join(out_folder, "code"), exist_ok=True)
        train_file = "train.py"
        os.system(f"cp {train_file} {code_dir}")
        os.system(f"cp -r networks {code_dir}")

    if world_size > 1:
        dist.barrier()

    global_step = 0
    first_epoch = 0

    if is_main_process:
        progress_bar = tqdm(range(global_step, max_train_steps))
        progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            # ============ 使用混合精度前向传播 ============
            with autocast("cuda", enabled=use_amp):
                # training
                pixel_values = tonemap(batch["gts"]).to(device) * 2 - 1
                LDR = batch["pixel_images"].to(device)
                EVS = batch["pixel_events"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                _batch_size = latents.shape[0]

                # 正确生成 timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (_batch_size,),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        [""] * _batch_size,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(latents.device)
                    encoder_hidden_states = text_encoder(prompt_ids)[0]

                # ============ 检查输入是否有 NaN（在DDP forward前，分布式同步） ============
                input_has_nan_tensor = torch.tensor(
                    [
                        torch.isnan(pixel_values).any()
                        or torch.isnan(LDR).any()
                        or torch.isnan(EVS).any()
                        or torch.isnan(latents).any()
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if world_size > 1:
                    dist.all_reduce(input_has_nan_tensor, op=dist.ReduceOp.MAX)

                input_has_nan = input_has_nan_tensor.item() > 0
                if input_has_nan:
                    if is_main_process:
                        print(
                            f"⚠️ Warning: Input has NaN at step {global_step}, skipping batch"
                        )
                    loss = torch.zeros(1, device=device, requires_grad=True)
                else:
                    # 获取条件特征
                    condition_images, condition_list = cond_encoder(LDR, EVS)

                    # ============ 关键：保护 condition_list ============
                    condition_list = safe_condition_list(condition_list)
                    # =================================================

                    # ControlNet 前向传播
                    down_block_res_samples, mid_block_res_samples = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=condition_list,
                        return_dict=False,
                    )

                    # UNet 前向传播
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_samples,
                    ).sample

                    # 计算目标
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # 计算损失
                    if config.get("use_snr_weighting", False):
                        # 使用SNR加权损失
                        snr_gamma = config.get("snr_gamma", 5.0)
                        snr_weighting_type = config.get("snr_weighting_type", "min_snr")
                        
                        # 计算每个timestep的SNR权重
                        snr_weight = compute_snr_weight(
                            timesteps, 
                            noise_scheduler, 
                            gamma=snr_gamma,
                            weighting_type=snr_weighting_type
                        )
                        
                        # 使用加权MSE损失
                        noise_loss_unweighted = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        # 对batch内每个样本应用权重
                        noise_loss = (noise_loss_unweighted * snr_weight[:, None, None, None]).mean()
                        unweighted_loss = noise_loss_unweighted.mean()
                        
                        # 记录SNR权重统计（用于监控）
                        if (not debug) and use_wandb and is_main_process and global_step % 100 == 0:
                            wandb.log({
                                "avg_snr_weight": snr_weight.mean().item(),
                                "noise_loss_unweighted": unweighted_loss.item(),
                            }, step=global_step)
                    else:
                        # 标准MSE损失（无加权）
                        noise_loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    
                    loss = noise_loss
            # ==========================================

            optimizer.zero_grad()

            # ============ 使用 scaler 反向传播 ============
            scaler.scale(loss).backward()

            # Unscale 梯度以便进行梯度裁剪
            scaler.unscale_(optimizer)

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            # ============ 检查梯度是否有 NaN（分布式同步） ============
            has_nan_grad = torch.tensor([0.0], device=device)
            for param in trainable_params:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = torch.tensor([1.0], device=device)
                        break

            if world_size > 1:
                dist.all_reduce(has_nan_grad, op=dist.ReduceOp.MAX)

            if has_nan_grad.item() > 0:
                if is_main_process:
                    print(
                        f"⚠️ Warning: NaN gradient detected at step {global_step}, skipping update"
                    )
                optimizer.zero_grad()
                scaler.update()
                continue
            # ==========================================

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            # ==========================================

            lr_scheduler.step()
            global_step += 1
            if is_main_process:
                progress_bar.update(1)

            # logging
            if (not debug) and use_wandb and is_main_process:
                wandb.log({"noise_loss": noise_loss.item()}, step=global_step)

            # saving
            if (
                global_step % checkpointing_steps == 0 or step == len(dataloader) - 1
            ) and is_main_process:
                save_path = os.path.join(out_folder, "checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict_controlnet": controlnet.module.state_dict()
                    if world_size > 1
                    else controlnet.state_dict(),
                    "state_dict_cond": cond_encoder.module.state_dict()
                    if world_size > 1
                    else cond_encoder.state_dict(),
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

                height = (
                    config.train_dataset.patch_size
                    if isinstance(config.train_dataset.patch_size, int)
                    else config.train_dataset.patch_size[0]
                )
                width = (
                    config.train_dataset.patch_size
                    if isinstance(config.train_dataset.patch_size, int)
                    else config.train_dataset.patch_size[1]
                )

                with torch.no_grad():
                    sample, results, latent = validation_pipeline(
                        prompt=encoder_hidden_states,
                        condition_images=batch["pixel_images"],
                        condition_events=batch["pixel_events"],
                        height=height,
                        width=width,
                        generators=generator,
                        **config.validation_setup,
                    )

                visuals = {}
                visuals[f"{global_step}_results_sample"] = (sample + 1) / 2
                visuals[f"{global_step}_events"] = batch["pixel_events"]
                visuals[f"{global_step}_images"] = batch["pixel_images"]
                visuals[f"{global_step}_gt"] = (pixel_values + 1) / 2
                save_path = os.path.join(out_folder, "images")
                save_image(visuals, save_path)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if is_main_process:
                progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    if is_main_process:
        print(f"\n{'=' * 60}")
        print("Training completed!")
        print(f"Total steps: {global_step}")
        print(f"Final checkpoint saved at: {out_folder}/checkpoints/")
        print(f"{'=' * 60}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default="")
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
        pretrained=args.pretrained,
    )
