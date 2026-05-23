# Batch Size修改说明

## 修改日期
2026-05-11

---

## 修改内容

### 配置文件修改：`config/Train_hybrid.yaml`

```yaml
# 修改前
batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.00005

# 修改后
batch_size: 2                    # ← 增大到2
gradient_accumulation_steps: 1   # ← 减少到1
learning_rate: 0.00005           # ← 保持不变
```

---

## 修改策略：方案A（保守方案）

### 为什么选择这个方案？

**核心思路**：保持有效batch size不变，只改变计算方式

**对比**：
```
修改前：
- batch_size: 1
- gradient_accumulation: 2
- 有效batch: 1 × 2 = 2
- 实际forward次数: 2次

修改后：
- batch_size: 2
- gradient_accumulation: 1
- 有效batch: 2 × 1 = 2
- 实际forward次数: 1次
```

**优势**：
- ✅ 有效batch size不变（仍然是2）
- ✅ Learning rate不需要调整（降低风险）
- ✅ 训练速度更快（减少1次forward pass）
- ✅ 最安全的方案，适合起步测试

---

## 参数影响分析

### 1. Learning Rate
**状态**: ✅ 保持不变

**原因**: 有效batch size不变，无需调整
- 原有效batch: 2
- 新有效batch: 2
- ratio: 1（无变化）
- lr: 保持5e-5

---

### 2. Gradient Accumulation
**状态**: ✅ 已调整（从2降到1）

**效果**:
- 减少梯度累积次数
- 每次更新只需1次forward，而非2次
- 训练速度提升约50%（理论上）

**计算**：
```
原配置：每步需要2次forward + 1次backward
新配置：每步需要1次forward + 1次backward
速度提升：(2+1)/(1+1) = 1.5倍
```

---

### 3. 其他参数
**状态**: ✅ 保持不变

```yaml
lr_warmup_steps: 500      # 不变
max_grad_norm: 1.0        # 不变
adam_weight_decay: 1.e-2  # 不变
adam_beta1: 0.99          # 不变
adam_beta2: 0.999         # 不变
```

**原因**: 有效batch不变，这些参数不需要调整

---

## 显存影响分析

### 预期显存占用

**batch_size=1时**：
- 约10-15GB（512×512分辨率）
- Gradient checkpointing已启用（节省显存）

**batch_size=2时**：
- 约20-25GB（增加约50-70%）
- 仍然在合理范围内（24GB GPU可承受）

**如果显存不足（OOM）**：
```yaml
# 方案1: 启用混合精度
修改train.py: use_amp = True

# 方案2: 回退到batch_size=1
修改config: batch_size: 1, gradient_accumulation_steps: 2
```

---

## 分布式训练影响

### 单GPU配置
```yaml
batch_size: 2
gradient_accumulation_steps: 1
world_size: 1
有效batch: 2 × 1 × 1 = 2
```

### 多GPU配置（例如4GPU）
```yaml
batch_size: 2（每GPU）
gradient_accumulation_steps: 1
world_size: 4
有效batch: 2 × 4 × 1 = 8

# 注意：代码中lr会乘以world_size
实际lr = 5e-5 × 4 = 2e-4
```

**说明**: 分布式训练时，有效batch增大了（2→8），但这是预期的，因为更多GPU需要更大的batch才能充分利用。

---

## 训练速度对比

### 理论计算

| 配置 | Forward次数 | Backward次数 | 每步耗时 | 总训练时间 |
|------|-----------|------------|---------|-----------|
| batch=1, accum=2 | 2次 | 1次 | ~1.0s | 200K步×1.0s=200Ks |
| batch=2, accum=1 | 1次 | 1次 | ~0.6s | 200K步×0.6s=120Ks |

**预期速度提升**: 约40-50%

**实际因素**：
- GPU利用率更高（batch=2时GPU更饱和）
- 减少Python overhead（更少的循环次数）
- 但forward耗时可能略微增加（处理更多数据）

---

## Wandb监控建议

### 需要观察的关键指标

**1. Loss曲线**
```
预期行为：
- noise_loss下降趋势应与之前类似
- 可能略微更平稳（更大的batch采样）
```

**2. 训练速度**
```
预期行为：
- 每秒处理的样本数应该增加
- 每步耗时应该减少
```

**3. 梯度统计**
```
预期行为：
- grad_norm应该保持稳定（仍被裁剪到1.0）
- 无NaN/Inf警告
```

**4. SNR加权指标**
```
预期行为：
- avg_snr_weight应该仍然在1.0-2.5范围
- noise_loss_unweighted对比值应该稳定
```

---

## 验证步骤

### Step 1: 检查配置修改
```bash
# 确认修改已生效
grep "batch_size" config/Train_hybrid.yaml
# 应输出: batch_size: 2

grep "gradient_accumulation_steps" config/Train_hybrid.yaml
# 应输出: gradient_accumulation_steps: 1
```

### Step 2: 测试显存
```bash
# 运行1个epoch测试显存是否足够
python train.py --config config/Train_hybrid.yaml --debug

# 观察显存占用
# 如果OOM，考虑启用混合精度或回退到batch=1
```

### Step 3: 开始训练
```bash
# 正式训练
python train.py --config config/Train_hybrid.yaml

# 监控wandb
# 观察loss曲线、训练速度、梯度范数
```

### Step 4: 对比效果
```bash
# 与之前batch_size=1的模型对比
# 重点对比：
# - 训练速度（应该更快）
# - PSNR（应该相似或更好）
# - 收敛速度（应该相似）
```

---

## 潜在问题和解决方案

### 问题1: 显存不足（OOM）

**症状**: 运行时报错 CUDA out of memory

**解决方案**:
```yaml
# 方案A: 启用混合精度（节省30-50%显存）
修改train.py line 74:
use_amp = True

# 方案B: 回退到batch_size=1
修改config:
batch_size: 1
gradient_accumulation_steps: 2

# 方案C: 降低图像分辨率
修改config:
patch_size: [256, 256]  # 从512降到256
```

---

### 问题2: 训练不稳定

**症状**: Loss震荡，NaN梯度

**原因**: 可能是SNR加权在新batch size下的数值问题

**解决方案**:
```yaml
# 方案A: 降低SNR gamma
snr_gamma: 3.0  # 从5.0降到3.0

# 方案B: 增大gradient clipping
max_grad_norm: 2.0  # 从1.0增大到2.0

# 方案C: 降低learning rate（保守）
learning_rate: 0.00003  # 从5e-5降到3e-5
```

---

### 问题3: 效果不如预期

**症状**: PSNR没有提升或反而下降

**原因**: batch size变化可能影响模型收敛

**解决方案**:
```yaml
# 方案A: 增加训练步数
max_train_steps: 300000  # 从200K增加到300K

# 方案B: 回退到batch_size=1（保守）
batch_size: 1
gradient_accumulation_steps: 2

# 方案C: 调整其他超参数
lr_scheduler_type: 'cosine'
```

---

## 下一步优化路径

### 如果batch_size=2效果好：
```yaml
# 可以尝试进一步增大到batch_size=4
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0001  # 需要增大2倍
max_grad_norm: 2.0
```

### 如果batch_size=2效果一般：
```yaml
# 保持batch_size=2，优化其他参数
learning_rate: 0.00008  # 微调lr
lr_scheduler_type: 'cosine'  # 使用cosine scheduler
adam_weight_decay: 0.02  # 微调weight decay
```

### 如果batch_size=2效果差：
```yaml
# 回退到batch_size=1
batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.00005
```

---

## 总结

### ✅ 修改完成
- batch_size: 1 → 2
- gradient_accumulation_steps: 2 → 1
- learning_rate: 保持不变

### ✅ 预期效果
- 训练速度提升40-50%
- 有效batch不变，训练行为相似
- 显存增加约50-70%（20-25GB）

### ⚠️ 注意事项
- 如果OOM，启用混合精度或回退
- 监控wandb指标变化
- 与之前模型对比效果

### 🎯 推荐流程
1. 测试显存是否足够
2. 开始训练
3. 监控wandb
4. 对比效果
5. 根据结果决定是否进一步调整

---

**状态**: ✅ 已修改，可开始训练  
**风险**: 低（有效batch不变）  
**预期**: 速度提升，效果相似或更好