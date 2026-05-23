# Batch Size调整快速参考

## 当前配置 → 新配置调整表

### 基线配置
```yaml
batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.00005
lr_warmup_steps: 500
max_grad_norm: 1.0

# 有效batch size:
#   单GPU: 1 × 1 × 2 = 2
#   4GPU:  1 × 4 × 2 = 8 (lr会×4 = 2e-4)
```

---

## 🎯 快速调整方案

### 方案A: 增大batch到2（推荐起步）

```yaml
batch_size: 2              # ← 改为2
gradient_accumulation_steps: 1  # ← 改为1
learning_rate: 0.00005     # ← 保持不变（有效batch仍为2）
lr_warmup_steps: 500       # ← 保持不变
max_grad_norm: 1.0         # ← 保持不变
```

**说明**: 
- 有效batch不变（2），lr不变
- 优点：训练速度更快（减少forward次数）
- 预期显存：~20-25GB

---

### 方案B: 增大batch到4（中等提升）

```yaml
batch_size: 4              # ← 改为4
gradient_accumulation_steps: 1  # ← 改为1
learning_rate: 0.0001      # ← 增大2倍 ⭐
lr_warmup_steps: 500       # ← 保持或增加到1000
max_grad_norm: 2.0         # ← 增大到2.0 ⭐
```

**说明**:
- 有效batch增大2倍（4），lr增大2倍
- 优点：训练稳定，收敛更快
- 预期显存：~35-45GB

**计算**:
- ratio = 4 / 2 = 2
- new_lr = 5e-5 × 2 = 1e-4

---

### 方案C: 增大batch到8（大幅提升）

```yaml
batch_size: 8              # ← 改为8
gradient_accumulation_steps: 1  # ← 改为1
learning_rate: 0.0002      # ← 增大4倍 ⭐
lr_warmup_steps: 250       # ← 减少到250（可选）
max_grad_norm: 5.0         # ← 增大到5.0 ⭐
adam_weight_decay: 0.05    # ← 增大weight decay ⭐
```

**说明**:
- 有效batch增大4倍（8），lr增大4倍
- 优点：最快训练，最高GPU利用率
- 预期显存：~60-80GB
- 风险：需要大显存，可能需要更多调优

**计算**:
- ratio = 8 / 2 = 4
- new_lr = 5e-5 × 4 = 2e-4

---

### 方案D: 启用混合精度 + batch增大

```yaml
batch_size: 4              # ← 改为4
gradient_accumulation_steps: 1
learning_rate: 0.0001
use_amp: True              # ← 启用混合精度 ⭐
```

**train.py修改**:
```python
use_amp = True  # 启用混合精度
```

**说明**:
- 混合精度节省显存30-50%
- 可支持更大batch size
- 需要测试数值稳定性（尤其是SNR加权）

---

## 📊 必须修改参数清单

### ⭐⭐⭐ 必须修改（否则会出问题）

| 参数 | 当前值 | batch=2时 | batch=4时 | batch=8时 |
|------|--------|----------|----------|----------|
| `batch_size` | 1 | 2 | 4 | 8 |
| `learning_rate` | 5e-5 | 5e-5 | **1e-4** | **2e-4** |
| `gradient_accumulation_steps` | 2 | 1 | 1 | 1 |

**关键**: Learning rate必须按Linear Scaling调整！

---

### ⭐⭐ 建议修改（提升效果）

| 参数 | 当前值 | batch增大时建议 | 原因 |
|------|--------|----------------|------|
| `max_grad_norm` | 1.0 | 2.0-5.0 | batch增大后梯度更大 |
| `lr_warmup_steps` | 500 | 500-1000 | 保持足够warmup样本数 |
| `adam_weight_decay` | 0.01 | 0.05-0.1 | 大batch需要更强regularization |

---

### ⭐ 可选修改（特殊场景）

| 参数 | 当前值 | 可选调整 | 适用场景 |
|------|--------|---------|---------|
| `lr_scheduler_type` | linear | cosine | 更好的收敛 |
| `adam_beta1` | 0.99 | 0.9 | 更快适应（可选） |
| `adam_beta2` | 0.999 | 0.95 | 可选调整 |

---

## ⚡ 快速修改命令

### 使用方案A（batch=2）
```bash
# 修改配置文件
sed -i 's/batch_size: 1/batch_size: 2/' config/Train_hybrid.yaml
sed -i 's/gradient_accumulation_steps: 2/gradient_accumulation_steps: 1/' config/Train_hybrid.yaml

# lr不变，其他不变
# 开始训练
python train.py --config config/Train_hybrid.yaml
```

### 使用方案B（batch=4，推荐）
```bash
# 修改配置文件
sed -i 's/batch_size: 1/batch_size: 4/' config/Train_hybrid.yaml
sed -i 's/gradient_accumulation_steps: 2/gradient_accumulation_steps: 1/' config/Train_hybrid.yaml
sed -i 's/learning_rate: 0.00005/learning_rate: 0.0001/' config/Train_hybrid.yaml
sed -i 's/max_grad_norm: 1.0/max_grad_norm: 2.0/' config/Train_hybrid.yaml

# 开始训练
python train.py --config config/Train_hybrid.yaml
```

### 使用方案C（batch=8）
```bash
# 修改配置文件
sed -i 's/batch_size: 1/batch_size: 8/' config/Train_hybrid.yaml
sed -i 's/gradient_accumulation_steps: 2/gradient_accumulation_steps: 1/' config/Train_hybrid.yaml
sed -i 's/learning_rate: 0.00005/learning_rate: 0.0002/' config/Train_hybrid.yaml
sed -i 's/max_grad_norm: 1.0/max_grad_norm: 5.0/' config/Train_hybrid.yaml
sed -i 's/adam_weight_decay: 1.e-2/adam_weight_decay: 0.05/' config/Train_hybrid.yaml

# 开始训练
python train.py --config config/Train_hybrid.yaml
```

---

## 🔍 分布式训练特殊考虑

### 多GPU时的lr计算

**代码中的处理**:
```python
# train.py line 240
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=config.learning_rate * world_size,  # ⭐ lr会乘以GPU数
    ...
)
```

**实际lr = config_lr × world_size**

**示例**（4GPU）:
```yaml
# 配置文件
learning_rate: 0.00005  # 基准lr

# 实际运行
lr = 0.00005 × 4 = 0.0002

# 有效batch size
effective_bs = batch_size × world_size × grad_accum
             = 1 × 4 × 2 = 8
```

### 多GPU batch调整

**情况1**: 保持每GPU batch不变，增大grad_accum
```yaml
# 修改前（4GPU）
batch_size: 1  # 每GPU
gradient_accumulation_steps: 2
# 有效batch: 1 × 4 × 2 = 8
# lr实际: 5e-5 × 4 = 2e-4

# 修改后
batch_size: 1  # 保持
gradient_accumulation_steps: 4  # 增大
# 有效batch: 1 × 4 × 4 = 16
# lr实际: 5e-5 × 4 × 2 = 4e-4（需手动调整config_lr）
```

**情况2**: 增大每GPU batch
```yaml
# 修改前（4GPU）
batch_size: 1
gradient_accumulation_steps: 2
# 有效batch: 8

# 修改后
batch_size: 2  # 增大
gradient_accumulation_steps: 1  # 减少
# 有效batch: 2 × 4 × 1 = 8（不变）
# lr实际: 5e-5 × 4 = 2e-4（不变）
```

**结论**: 多GPU时，如果有效batch不变，lr不需要调整！

---

## 🎯 我的推荐方案

### 如果显存足够（> 40GB）
**推荐方案B**: batch_size=4
```yaml
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0001
max_grad_norm: 2.0
lr_warmup_steps: 500
```

**理由**:
- 有效batch增大2倍（4），稳定训练
- lr增大2倍，收敛更快
- 显存占用适中
- 预期效果提升明显

---

### 如果显存中等（24-40GB）
**推荐方案A**: batch_size=2
```yaml
batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 0.00005  # 不变
max_grad_norm: 1.0
lr_warmup_steps: 500
```

**理由**:
- 有效batch不变（2），lr不变，风险小
- 训练速度提升（减少forward次数）
- 显存占用可控

---

### 如果显存不足（< 24GB）
**推荐**: 保持batch_size=1，或启用混合精度
```yaml
batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.00005
use_amp: True  # 启用混合精度
```

---

## ✅ 修改后验证清单

### 训练前检查
- [ ] 修改了batch_size
- [ ] 按Linear Scaling调整了learning_rate
- [ ] 调整了gradient_accumulation_steps
- [ ] 调整了max_grad_norm（可选）
- [ ] 检查显存是否足够（运行1个batch测试）
- [ ] 检查lr_warmup_steps是否合理

### 训练时监控
- [ ] Wandb: 查看loss曲线是否平稳
- [ ] Wandb: 查看梯度范数是否正常
- [ ] 检查是否有NaN/Inf警告
- [ ] 监控显存占用是否稳定
- [ ] 检查训练速度是否提升

### 效果对比
- [ ] 与原配置对比PSNR/SSIM
- [ ] 对比训练收敛速度
- [ ] 对比最终模型质量

---

**总结**: 
- **必须修改**: batch_size, lr（按比例），gradient_accumulation
- **建议修改**: max_grad_norm, lr_warmup_steps
- **可选修改**: weight_decay, scheduler, optimizer

**推荐**: 方案B (batch_size=4) 为最佳平衡点