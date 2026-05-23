"""测试SNR加权函数的正确性"""

import torch
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler

def compute_snr_weight(timesteps, noise_scheduler, gamma=5.0, weighting_type="min_snr"):
    """计算SNR加权系数"""
    alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    if weighting_type == "min_snr":
        snr_clipped = torch.clamp(snr, min=1.0, max=gamma)
        weight = torch.sqrt(snr_clipped)
    elif weighting_type == "inverse":
        weight = 1.0 / (snr + 1e-8)
        weight = torch.clamp(weight, min=0.1, max=10.0)
    elif weighting_type == "sqrt":
        weight = torch.sqrt(snr + 1e-8)
    else:
        raise ValueError(f"Unknown weighting_type: {weighting_type}")
    
    return weight


def visualize_snr_weighting():
    """可视化不同SNR加权策略"""
    
    # 创建噪声调度器（SD默认参数）
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon"
    )
    
    # 所有timesteps
    all_timesteps = torch.arange(0, 1000)
    
    # 计算SNR值
    alphas_cumprod = noise_scheduler.alphas_cumprod[all_timesteps]
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    # 计算三种加权策略
    weight_min_snr = compute_snr_weight(all_timesteps, noise_scheduler, gamma=5.0, weighting_type="min_snr")
    weight_inverse = compute_snr_weight(all_timesteps, noise_scheduler, weighting_type="inverse")
    weight_sqrt = compute_snr_weight(all_timesteps, noise_scheduler, weighting_type="sqrt")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: SNR值分布
    axes[0, 0].plot(all_timesteps.numpy(), snr.numpy(), label='SNR', color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('SNR')
    axes[0, 0].set_title('Signal-to-Noise Ratio (SNR) across Timesteps')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 子图2: Min-SNR weighting (Stable Diffusion使用)
    axes[0, 1].plot(all_timesteps.numpy(), weight_min_snr.numpy(), 
                    label='Min-SNR Weight (gamma=5.0)', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_title('Min-SNR Weighting (Stable Diffusion)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: Inverse SNR weighting
    axes[1, 0].plot(all_timesteps.numpy(), weight_inverse.numpy(), 
                    label='Inverse SNR Weight', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Inverse SNR Weighting (偏向高噪声)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 所有加权策略对比
    axes[1, 1].plot(all_timesteps.numpy(), weight_min_snr.numpy(), 
                    label='Min-SNR (SD)', color='green', linewidth=2)
    axes[1, 1].plot(all_timesteps.numpy(), weight_inverse.numpy(), 
                    label='Inverse', color='orange', linewidth=2)
    axes[1, 1].plot(all_timesteps.numpy(), weight_sqrt.numpy(), 
                    label='Sqrt', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_title('Weighting Strategies Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('snr_weighting_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化结果已保存到 snr_weighting_analysis.png")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("SNR加权统计信息")
    print("="*60)
    print(f"低噪声时刻 (t=0-200):")
    print(f"  - 平均SNR: {snr[:200].mean().item():.2f}")
    print(f"  - Min-SNR权重: {weight_min_snr[:200].mean().item():.3f}")
    print(f"\n中等噪声时刻 (t=400-600):")
    print(f"  - 平均SNR: {snr[400:600].mean().item():.2f}")
    print(f"  - Min-SNR权重: {weight_min_snr[400:600].mean().item():.3f}")
    print(f"\n高噪声时刻 (t=800-999):")
    print(f"  - 平均SNR: {snr[800:].mean().item():.4f}")
    print(f"  - Min-SNR权重: {weight_min_snr[800:].mean().item():.3f}")
    print("="*60)
    
    # 解释效果
    print("\n💡 Min-SNR加权的效果解释:")
    print("- 低噪声时刻(t小) → 高SNR → 较大权重")
    print("  → 模型会更关注低噪声预测的准确性")
    print("- 高噪声时刻(t大) → 低SNR → 权重被限制到1.0")
    print("  → 避免在极高噪声时梯度信号过弱")
    print("- gamma=5.0限制最大SNR，防止权重过大")
    print("  → 这是Stable Diffusion训练中使用的标准参数")


def test_snr_weight_gradient():
    """测试SNR加权对梯度的影响"""
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon"
    )
    
    # 模拟不同timestep的损失
    batch_size = 4
    timesteps = torch.tensor([50, 300, 600, 900])  # 低、中、高噪声
    
    # 假设每个样本的基础损失都是1.0
    base_loss = torch.ones(batch_size, 1, 1, 1)
    
    # 计算加权损失
    snr_weight = compute_snr_weight(timesteps, noise_scheduler, gamma=5.0)
    weighted_loss = (base_loss * snr_weight[:, None, None, None]).mean()
    
    print("\n" + "="*60)
    print("梯度影响测试")
    print("="*60)
    print(f"Timesteps: {timesteps.tolist()}")
    print(f"SNR weights: {snr_weight.tolist()}")
    print(f"Weighted total loss: {weighted_loss.item():.3f}")
    print("="*60)
    print("\n解释:")
    print("- timestep=50 (低噪声): 权重最大 → 梯度贡献最大")
    print("- timestep=900 (高噪声): 权重最小 → 梯度贡献最小")
    print("→ 模型会优先学习低噪声时刻的准确预测")


if __name__ == "__main__":
    print("="*60)
    print("SNR加权损失测试")
    print("="*60)
    
    visualize_snr_weighting()
    test_snr_weight_gradient()
    
    print("\n✅ 测试完成！")
    print("\n建议:")
    print("1. 使用 Min-SNR weighting (gamma=5.0)")
    print("2. 这是Stable Diffusion验证有效的策略")
    print("3. 可以根据实际效果调整gamma值（通常在3-10之间）")