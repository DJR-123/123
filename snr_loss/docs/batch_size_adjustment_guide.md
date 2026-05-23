# 调整Batch Size同步修改指南

## 当前配置分析

### 现有参数
```yaml
batch_size: 1              # 每个GPU的batch size
gradient_accumulation_steps: 2  # 梯度累积步数
learning_rate: 0.00005     # 学习率 5e-5
lr_warmup_steps: 500       # warmup步数
world_size: N              # GPU数量（运行时确定）
```

### 有效Batch Size计算
```
Effective Batch Size = batch_size × world_size × gradient_accumulation_steps

当前情况：
- 单GPU: 1 × 1 × 2 = 2
- 4GPU:  1 × 4 × 2 = 8
- 8GPU:  1 × 8 × 2 = 16
```

---

## 🎯 调整Batch Size的策略

### 方案1: 增大单GPU batch_size（推荐）

**场景**: 显存足够，想要更快训练

**示例**: 从batch_size=1增大到4

**需要修改**:

#### 1. **Learning Rate调整** ⭐⭐⭐ 最重要

**Linear Scaling Rule**: batch size增大k倍，lr也增大k倍

```yaml
# 修改前
batch_size: 1
gradient_accumulation_steps: 2
effective_batch_size: 2 (单GPU) 或 8 (4GPU)
learning_rate: 0.00005

# 修改后（batch_size增大4倍）
batch_size: 4
gradient_accumulation_steps: 1  # 减少累积步数
effective_batch_size: 4 (单GPU) 或 16 (4GPU)
learning_rate: 0.00005 × 2 = 0.0001  # 增大2倍
```

**计算公式**:
```python
# 新lr = 基准lr × (新有效batch_size / 原有效batch_size)
new_lr = 5e-5 * (new_effective_bs / old_effective_bs)
```

**示例计算**:
- 原有效batch size: 2 (单GPU)
- 新有效batch size: 4 (单GPU, batch_size=4, grad_accum=1)
- ratio = 4/2 = 2
- new_lr = 5e-5 × 2 = 1e-4

---

#### 2. **Gradient Accumulation调整**

**原则**: 保持有效batch size合理

```yaml
# 方案A: 增大batch_size，减少accumulation
batch_size: 4
gradient_accumulation_steps: 1
# 有效batch = 4，lr增大到1e-4

# 方案B: 增大batch_size，保持accumulation（有效batch更大）
batch_size: 2
gradient_accumulation_steps: 2
# 有效batch = 4，lr增大到1e-4

# 方案C: 只增大batch_size，保持accumulation
batch_size: 4
gradient_accumulation_steps: 2
# 有效batch = 8，lr增大到2e-4
```

**建议**: 减少accumulation，增大实际batch_size（训练更快）

---

#### 3. **Warmup Steps调整**

**原则**: warmup步数应覆盖足够的有效batch数量

```yaml
# 修改前
lr_warmup_steps: 500  # warmup 500步

# 修改后（batch增大后，warmup可适当增加）
lr_warmup_steps: 1000  # 或保持500
```

**经验法则**:
- Warmup覆盖的样本数应足够（至少几千到几万样本）
- Warmup步数 × 有效batch_size = warmup覆盖样本数
- 如果有效batch增大2倍，warmup步数可减半或保持不变

**示例**:
```
原配置: 500步 × 2样本 = 1000样本warmup
新配置: 250步 × 4样本 = 1000样本warmup（相同）
或    : 500步 × 4样本 = 2000样本warmup（更安全）
```

---

#### 4. **其他参数考虑**

**Weight Decay**: 通常不需要调整
```yaml
adam_weight_decay: 1.e-2  # 保持不变
```

**Gradient Clipping**: 可能需要调整
```yaml
max_grad_norm: 1.0  # batch增大后，梯度累积更多，可能需要适当增大
# 可尝试: max_grad_norm: 2.0 或 5.0
```

**Adam Betas**: 通常不需要调整
```yaml
adam_beta1: 0.99
adam_beta2: 0.999
```

---

### 方案2: 保持batch_size，增大gradient_accumulation

**场景**: 显存受限，无法增大实际batch size

**示例**: gradient_accumulation从2增大到4

```yaml
# 修改前
batch_size: 1
gradient_accumulation_steps: 2
effective_batch_size: 2
learning_rate: 0.00005

# 修改后
batch_size: 1
gradient_accumulation_steps: 4
effective_batch_size: 4
learning_rate: 0.00005 × 2 = 0.0001
```

**优点**: 
- 显存占用不变
- 有效batch增大，训练更稳定

**缺点**: 
- 训练速度变慢（更多forward pass）

---

## 📊 不同Batch Size配置对比

### 配置对比表

| 配置 | batch_size | grad_accum | world_size | 有效batch | 推荐lr | 训练速度 | 显存 |
|------|-----------|-----------|-----------|----------|--------|---------|------|
| 原配置 | 1 | 2 | 1 | 2 | 5e-5 | 慢 | 低 |
| 方案A | 2 | 1 | 1 | 2 | 5e-5 | 快 | 中 |
| 方案B | 4 | 1 | 1 | 4 | 1e-4 | 最快 | 高 |
| 方案C | 1 | 4 | 1 | 4 | 1e-4 | 最慢 | 低 |
| 方案D | 4 | 2 | 1 | 8 | 2e-4 | 快 | 最高 |

### 推荐配置（渐进式）

#### 第一阶段: 中等batch（显存允许）
```yaml
batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 0.00005  # 有效batch不变，lr不变
lr_warmup_steps: 500
max_grad_norm: 1.0
```
**有效batch**: 2 (单GPU) 或 8 (4GPU)

#### 第二阶段: 更大batch（显存充足）
```yaml
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0001  # 增大2倍
lr_warmup_steps: 500
max_grad_norm: 2.0
```
**有效batch**: 4 (单GPU) 或 16 (4GPU)

#### 第三阶段: 最大batch（显存很大）
```yaml
batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 0.0002  # 增大4倍
lr_warmup_steps: 250  # 减少warmup步数
max_grad_norm: 5.0
```
**有效batch**: 8 (单GPU) 或 32 (4GPU)

---

## ⚠️ 注意事项和风险

### 1. 显存占用增加

**预估显存需求**:
```
Batch=1: 约10-15GB (512×512图像)
Batch=2: 约20-25GB
Batch=4: 约35-45GB
Batch=8: 约60-80GB

具体占用取决于：
- 图像分辨率（512×512）
- 模型架构（ControlNet + Encoder + UNet部分）
- Gradient checkpointing（已启用，节省显存）
```

**检查方法**:
```python
# 训练开始时查看显存占用
import torch
print(torch.cuda.memory_allocated() / 1024**3)  # GB
print(torch.cuda.max_memory_allocated() / 1024**3)  # GB
```

---

### 2. Learning Rate调整的风险

**风险1**: lr过大导致不稳定
- **现象**: Loss震荡，NaN梯度，收敛失败
- **解决**: 降低lr或增大warmup

**风险2**: lr过小导致收敛慢
- **现象**: Loss下降缓慢，需要更多训练步数
- **解决**: 增大lr或延长训练

**安全策略**:
```yaml
# 渐进式调整
Round 1: batch=2, lr=5e-5（保守，先测试）
Round 2: batch=4, lr=8e-5（逐步增大）
Round 3: batch=4, lr=1e-4（最终目标）
```

---

### 3. 大Batch训练的特殊考虑

**现象**: 大batch训练可能收敛到"尖锐"极小值，泛化性差

**解决方案**:
```yaml
# 方案A: 增大weight decay
adam_weight_decay: 0.1  # 从0.01增大到0.1

# 方案B: 使用SAM优化器（Sharpness-Aware Minimization）
optimizer: SAM
rho: 0.05

# 方案C: 保持warmup充足
lr_warmup_steps: 1000  # 增大warmup

# 方案D: 添加lr decay
lr_scheduler_type: 'cosine'  # 使用cosine衰减
```

---

## 🔧 实施步骤

### Step 1: 测试显存
```bash
# 先用batch_size=2测试
修改config: batch_size: 2
运行训练，观察显存占用

如果OOM（Out of Memory）:
  → 降低batch_size或启用混合精度

如果显存充足:
  → 继续增大到batch_size=4
```

### Step 2: 调整Learning Rate
```yaml
# 根据有效batch size调整
原有效batch = 1 × world_size × 2
新有效batch = 新batch_size × world_size × 新grad_accum

new_lr = 5e-5 × (新有效batch / 原有效batch)
```

### Step 3: 调整其他参数
```yaml
# 根据情况调整
lr_warmup_steps: 500-1000
max_grad_norm: 1.0-5.0
adam_weight_decay: 0.01-0.1
```

### Step 4: 监控训练
```bash
# Wandb监控关键指标
- loss下降是否平稳
- 是否有NaN/Inf
- 梯度范数是否稳定
- 验证集效果是否改善
```

---

## 💡 具体修改示例

### 示例1: 单GPU，batch增大到4

**修改前**:
```yaml
batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.00005
lr_warmup_steps: 500
max_grad_norm: 1.0
adam_weight_decay: 1.e-2
```

**修改后**:
```yaml
batch_size: 4
gradient_accumulation_steps: 1  # 减少累积
learning_rate: 0.0001  # 增大2倍
lr_warmup_steps: 500  # 保持不变
max_grad_norm: 2.0  # 稍微增大
adam_weight_decay: 1.e-2  # 保持不变
```

**计算**:
- 原有效batch: 1 × 1 × 2 = 2
- 新有效batch: 4 × 1 × 1 = 4
- ratio: 4/2 = 2
- new_lr: 5e-5 × 2 = 1e-4

---

### 示例2: 4GPU，保持有效batch不变

**修改前**:
```yaml
batch_size: 1  # 每GPU
gradient_accumulation_steps: 2
# 有效batch: 1 × 4 × 2 = 8
learning_rate: 0.00005 × 4 = 0.0002  # (假设代码中lr *= world_size)
```

**修改后**:
```yaml
batch_size: 2  # 每GPU
gradient_accumulation_steps: 1
# 有效batch: 2 × 4 × 1 = 8（不变）
learning_rate: 0.00005 × 4 = 0.0002  # 不变
```

**说明**: 分布式训练时，`lr = base_lr × world_size`，增大batch_size后有效batch不变则lr不变

---

### 示例3: 启用混合精度以支持更大batch

**修改**:
```yaml
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0001
```

**train.py修改**:
```python
use_amp = True  # 启用混合精度（节省显存30-50%）
```

**效果**: 显存减少，可支持batch_size=4甚至更大

---

## 📈 性能影响分析

### 训练速度对比

| Batch Size | 每步耗时 | 总步数 | 总训练时间 | 收敛质量 |
|-----------|---------|--------|-----------|---------|
| 1 (accum=2) | 0.5s | 200K | 100K秒 | 基线 |
| 2 (accum=1) | 0.6s | 200K | 120K秒 | 更快 |
| 4 (accum=1) | 0.8s | 200K | 160K秒 | 最快 |

**注意**: batch增大后，可能需要更多步数才能收敛到相同质量

---

### Batch Size对训练的影响

**大Batch优势**:
- ✅ 训练更稳定（梯度估计更准确）
- ✅ 可以使用更大lr（加速训练）
- ✅ GPU利用率更高

**大Batch劣势**:
- ⚠️ 显存占用大
- ⚠️ 可能收敛到尖锐极小值（泛化性差）
- ⚠️ 需要调整lr和其他超参数

**最佳实践**:
- 有效batch size推荐范围: 16-64
- 太小(< 8): 训练不稳定
- 太大(> 128): 泛化性可能下降

---

## ✅ 快速配置模板

### 模板1: 小显存GPU (< 24GB)
```yaml
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 0.0001
lr_warmup_steps: 1000
max_grad_norm: 1.0
# 有效batch: 4，显存占用低，速度慢
```

### 模板2: 中等显存GPU (24-40GB)
```yaml
batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 0.0001
lr_warmup_steps: 500
max_grad_norm: 2.0
# 有效batch: 4，平衡方案
```

### 模板3: 大显存GPU (> 40GB)
```yaml
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0001
lr_warmup_steps: 500
max_grad_norm: 2.0
# 有效batch: 4，速度快
```

### 模板4: 多GPU (4卡)
```yaml
batch_size: 2  # 每GPU
gradient_accumulation_steps: 1
learning_rate: 0.00005  # 基准lr（代码中会×world_size）
lr_warmup_steps: 500
max_grad_norm: 2.0
# 有效batch: 2 × 4 × 1 = 8，最终lr = 2e-4
```

---

## 🎯 总结建议

### 必须同步修改 ⭐⭐⭐
1. **Learning Rate**: Linear scaling，batch增大k倍，lr增大k倍
2. **Gradient Accumulation**: 调整以保持合理有效batch

### 建议同步修改 ⭐⭐
3. **Warmup Steps**: batch增大后可适当调整
4. **Gradient Clipping**: batch增大后梯度可能更大，调整max_grad_norm

### 可选修改 ⭐
5. **Weight Decay**: 大batch可能需要增大weight decay
6. **Optimizer**: 可考虑SAM优化器
7. **Scheduler**: 使用cosine scheduler

### 测试流程
1. 先测试显存是否足够
2. 按Linear scaling调整lr
3. 监控loss曲线和梯度
4. 根据效果微调参数
5. 对比不同batch size的效果

---

**核心原则**: 
- **Linear Scaling Rule**: lr ∝ batch_size
- **显存优先**: 根据GPU显存决定batch_size
- **渐进调整**: 不要一次性改动太大
- **监控验证**: 密切关注训练过程和最终效果