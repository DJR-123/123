# 涨点建议总结

## 当前项目分析

### 架构
- **Base Model**: Stable Diffusion 1.5 + ControlNet (IP2P)
- **Condition Encoder**: Hybrid (ConvLSTM + Swin Transformer)
- **Training Strategy**: 
  - UNet: 只训练 `conv_out`
  - ControlNet: 全部训练
  - Encoder: 全部训练

### 当前配置
- Learning Rate: 5e-5
- Batch Size: 1 (x2 gradient accumulation = effective batch 2)
- Optimizer: AdamW (beta1=0.99, beta2=0.999, weight_decay=0.01)
- Scheduler: Linear with 500 warmup steps
- Max Steps: 200K
- Precision: FP32 (AMP disabled)

---

## 🎯 涨点建议（按优先级排序）

### ⭐⭐⭐ 高优先级（预期收益大）

#### 1. **数据增强优化**
**当前问题**: 数据增强较弱，只有random crop和LDR生成

**建议改进**:
```python
# 在 data_processing/evs_image_h5_dataset.py 添加：
- 随机水平/垂直翻转 (p=0.5)
- 随机旋转 (90/180/270度)
- 色彩抖动 (亮度/对比度/饱和度，轻微)
- 随机裁剪时加入尺度变化 (scale: 0.8-1.2)
- CutMix/MixUp (可选，针对HDR数据可能需要特殊处理)
```

**预期收益**: +0.5-1.0 dB PSNR  
**实现难度**: 低  
**风险**: 低

---

#### 2. **学习率策略优化**
**当前问题**: Linear scheduler，可能收敛过快或不充分

**建议改进**:
```yaml
# 方案A: Cosine Annealing with Warm Restarts
lr_scheduler_type: 'cosine_with_restarts'
num_cycles: 3  # 200K steps分成3个周期

# 方案B: One-cycle policy（通常效果更好）
lr_scheduler_type: 'cosine'
max_lr: 0.0001  # 峰值lr
div_factor: 25  # 初始lr = max_lr/25
final_div_factor: 100  # 最终lr = max_lr/100

# 方案C: 分阶段训练
# Stage 1 (0-100K): lr=5e-5, 训练所有模块
# Stage 2 (100K-150K): lr=1e-5, 冻结encoder，只训练ControlNet+UNet
# Stage 3 (150K-200K): lr=5e-6, 全模型微调
```

**预期收益**: +0.3-0.8 dB PSNR  
**实现难度**: 低  
**风险**: 低

---

#### 3. **增加训练步数 + 模型保存策略**
**当前问题**: 200K步可能不够充分训练，且缺少best model保存

**建议改进**:
```yaml
# 增加训练步数
max_train_steps: 400000  # 或更多

# 添加validation-based checkpointing
checkpointing_steps: 5000
validation_steps: 2000
save_best_only: True  # 保存验证集效果最好的模型
keep_last_n_checkpoints: 3
```

**实现要点**:
- 添加验证集评估逻辑（PSNR/SSIM/LPIPS）
- 根据验证集指标保存best checkpoint
- 学习曲线监控，判断是否过拟合/欠拟合

**预期收益**: +0.5-1.5 dB PSNR（如果当前欠拟合）  
**实现难度**: 中  
**风险**: 低

---

#### 4. **SNR加权损失** ⭐ 推荐
**当前问题**: 对所有timestep使用相同权重，忽略了不同噪声水平的重要性差异

**建议改进**:
```python
# 在 train.py 中实现SNR加权
def get_snr_weight(timesteps, noise_scheduler, gamma=5.0):
    """SNR加权：低噪声（高SNR）时权重更大
    
    Args:
        gamma: 加权强度，越大越偏向低噪声
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    # 方法1: Min-SNR weighting (used in SD training)
    snr = torch.clamp(snr, min=1.0, max=5.0)
    weight = torch.sqrt(snr)
    
    # 方法2: Inverse SNR (偏向高噪声)
    # weight = 1.0 / (snr + 1e-8)
    
    # 方法3: 自适应加权
    # weight = torch.pow(snr, gamma)
    
    return weight

# 修改训练循环
snr_weight = get_snr_weight(timesteps, noise_scheduler)
noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
noise_loss = (noise_loss * snr_weight).mean()
```

**预期收益**: +0.3-1.0 dB PSNR  
**实现难度**: 低  
**风险**: 低  
**参考文献**: 
- "Efficient Diffusion Training via Min-SNR Weighting Strategy" (Hang et al., 2023)
- Stable Diffusion官方训练也使用了SNR加权

---

### ⭐⭐ 中优先级（值得尝试）

#### 5. **EMA (Exponential Moving Average)**
**当前问题**: 训练不稳定，模型权重波动

**建议改进**:
```python
from torch_ema import ExponentialMovingAverage

# 初始化EMA
ema = ExponentialMovingAverage(
    trainable_params, 
    decay=0.9999
)

# 训练循环中
with ema.average_parameters():
    # 使用EMA参数进行验证
    validate(...)
    
# 推理时使用EMA参数
ema.copy_to(model.parameters())
```

**预期收益**: +0.2-0.5 dB PSNR  
**实现难度**: 低  
**风险**: 低

---

#### 6. **混合精度训练 (AMP)**
**当前问题**: 使用FP32训练，显存占用大，batch size受限

**建议改进**:
```python
# 在 train.py 中启用
use_amp = True  # 启用混合精度

# 修改前向传播
with autocast("cuda", enabled=use_amp):
    # 前向传播
    model_pred = unet(...)

# 修改损失计算（保持精度）
noise_loss = F.mse_loss(
    model_pred.float(),  # 转回FP32计算loss
    target.float(),
    reduction='mean'
)
```

**优势**:
- 显存减少 ~30-50%
- 可以增大batch size到2-4
- 训练速度提升 ~20-30%

**预期收益**: +0.2-0.5 dB PSNR（通过更大batch size）  
**实现难度**: 低  
**风险**: 低（需要测试数值稳定性）

---

#### 7. **优化器超参数调优**
**当前问题**: AdamW beta参数可能不是最优

**建议改进**:
```yaml
# 方案A: 标准AdamW参数（更稳定）
adam_beta1: 0.9  # 当前是0.99，偏高
adam_beta2: 0.999
adam_weight_decay: 0.01

# 方案B: 对于大模型训练（推荐）
adam_beta1: 0.9
adam_beta2: 0.95  # 降低beta2，更快适应
adam_weight_decay: 0.1  # 增大weight decay

# 方案C: SAM优化器（Sharpness-Aware Minimization）
# 寻找更平坦的极小值，泛化性更好
optimizer: SAM
rho: 0.05
```

**预期收益**: +0.1-0.3 dB PSNR  
**实现难度**: 低  
**风险**: 低

---

#### 8. **微调更多UNet层**
**当前问题**: 只训练 `conv_out`，可能限制了表达能力

**建议改进**:
```yaml
# 分阶段策略
# Stage 1 (0-100K): 只训练 conv_out
unet_trainable_modules:
  - "conv_out"

# Stage 2 (100K-200K): 解冻更多层
unet_trainable_modules:
  - "conv_out"
  - "up_blocks.3"  # 最高分辨率的上采样块
  - "up_blocks.2"

# Stage 3 (200K+): 全模型微调（低学习率）
train_unet: True
unet_trainable_modules:
  - ""  # 空字符串表示全部可训练
learning_rate: 1e-6  # 更低的学习率
```

**预期收益**: +0.3-0.8 dB PSNR  
**实现难度**: 中  
**风险**: 中（可能过拟合，需要监控）

---

#### 9. **噪声调度器优化**
**当前问题**: 使用默认SD调度器，可能不是HDR最优

**建议改进**:
```yaml
# 方案A: v-prediction（更适合生成任务）
noise_scheduler_kwargs:
  prediction_type: "v_prediction"
  beta_schedule: "scaled_linear"
  rescale_betas_zero_snr: True  # 重要！配合v-prediction

# 方案B: 更激进的噪声调度（HDR需要保留细节）
beta_start: 0.0005  # 降低初始噪声
beta_end: 0.015     # 提高最终噪声

# 方案C: 使用零终端SNR（Zero Terminal SNR）
rescale_betas_zero_snr: True
prediction_type: "v_prediction"
```

**预期收益**: +0.2-0.5 dB PSNR  
**实现难度**: 中  
**风险**: 中（需要重新训练）

---

### ⭐ 低优先级（长期优化）

#### 10. **ControlNet架构改进**
**建议改进**:
- 尝试不同的ControlNet变体
- 添加更多condition注入点
- 使用更强的特征融合机制

**预期收益**: +0.2-0.5 dB PSNR  
**实现难度**: 高  
**风险**: 高

---

#### 11. **Encoder架构改进**
**建议改进**:
- 增加Swin Transformer的深度/宽度
- 尝试更先进的融合策略（cross-attention）
- 添加多尺度特征聚合

**预期收益**: +0.3-0.7 dB PSNR  
**实现难度**: 高  
**风险**: 中

---

#### 12. **后处理优化**
**建议改进**:
- 训练一个轻量级的refinement网络
- 使用ESRGAN进行超分辨率
- HDR色调映射优化

**预期收益**: +0.3-0.8 dB PSNR  
**实现难度**: 中  
**风险**: 低

---

## 🚀 快速实验建议（按实现难度排序）

### Phase 1: 配置调优（1-2周）
1. ✅ 启用混合精度训练（增大batch size）
2. ✅ 修改学习率调度器（Cosine with restarts）
3. ✅ 添加SNR加权损失
4. ✅ 增加数据增强

### Phase 2: 训练策略优化（2-4周）
1. ✅ 实现EMA
2. ✅ 增加训练步数 + validation checkpointing
3. ✅ 分阶段训练策略
4. ✅ 尝试v-prediction

### Phase 3: 架构优化（1-2月）
1. 🔬 微调更多UNet层
2. 🔬 改进Encoder融合策略
3. 🔬 添加后处理网络

---

## 📊 评估指标建议

### 定量指标
- PSNR (主要)
- SSIM (结构相似度)
- LPIPS (感知质量，可选)
- HDR-VDP (HDR专用，如果可用)

### 定性评估
- 视觉质量对比
- 细节保留程度
- 过曝/欠曝区域恢复
- 色彩准确性

### 验证集设置
- 使用独立验证集（非训练集）
- 定期验证（每1000-2000步）
- 保存best checkpoint

---

## 🛠️ 实验记录模板

```markdown
## 实验名称: [日期]_[实验内容]
- 基线: PSNR=XX.XX dB
- 修改内容:
  1. ...
  2. ...
- 训练配置:
  - lr: 
  - batch_size:
  - scheduler:
  - steps:
- 结果:
  - PSNR: XX.XX dB (+/- X.XX)
  - 训练时间:
  - 显存占用:
- 分析:
  - ...
- 下一步:
  - ...
```

---

## 💡 额外建议

### 调试技巧
1. **梯度监控**: 定期检查梯度范数、分布
2. **学习曲线**: 使用wandb监控loss、grad_norm等
3. **消融实验**: 每次只改一个变量
4. **Early stopping**: 如果连续X次验证无提升，停止

### 代码优化
1. **添加logging**: 记录每个模块的梯度、激活值统计
2. **可视化工具**: 定期可视化latent、attention map等
3. **单元测试**: 对关键模块添加测试

### 资源优化
1. **Gradient checkpointing**: 已启用 ✓
2. **Mixed precision**: 建议启用
3. **Gradient accumulation**: 已使用 ✓
4. **Dataloader优化**: 增加`pin_memory=True`, `prefetch_factor`

---

## 📚 参考资源

### Diffusion Model训练技巧
- [Stable Diffusion Training Tips](https://huggingface.co/docs/diffusers/training)
- [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)
- [v-prediction parameterization](https://arxiv.org/abs/2202.00512)

### HDR重建相关
- HDRev: HDR Video Reconstruction
- Event-based HDR reconstruction papers

### 优化器相关
- [SAM Optimizer](https://arxiv.org/abs/2010.01412)
- [AdamW vs Adam](https://arxiv.org/abs/1711.05101)

---

**文档维护**: 建议每次实验后更新结果，记录什么有效、什么无效，逐步积累经验。