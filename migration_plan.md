# GeoReward 功能移植方案：now → now_cosmos3（v2）

## 概述

将 `now` 项目（Wan2.2 + 4RC）中三个未移植的高级功能集成到 `now_cosmos3`（Cosmos3 + 4RC）项目中：

| 功能 | 节省算力 | 核心思路 |
|------|---------|---------|
| 渐进淘汰 BoN | ~40% | 在去噪中间 checkpoint 提前淘汰低分候选 |
| 树分支 BoN | ~29%（相对渐进） | 少量 trunk 共享前段去噪 + 分支点注入噪声 |
| 梯度引导 | — | 去噪过程中反向传播 4RC loss 引导 latent |

## 核心挑战

`now` 项目中这三个功能深度依赖 Wan2.2 的 `WanI2V` 类暴露的低层 API：

- `prepare_progressive()` — 批量准备候选初始状态（**每候选独立 scheduler 实例**）
- `denoise_candidates()` — 按步去噪（可控 start/end step）
- `denoise_candidates_with_guidance()` — 带梯度引导的逐步去噪
- `extract_pred_x0()` — 从中间步提取预测的干净 latent
- `find_step_for_sigma()` — sigma → step 映射
- `branch_candidates()` — trunk latent 克隆 + **深拷贝 scheduler** + 噪声注入
- `decode_latent()` — VAE 解码
- `cleanup_progressive()` — 内存清理

而 `now_cosmos3` 使用 HuggingFace Diffusers 的 `Cosmos3OmniPipeline`，其 `__call__` 方法是一个黑盒（一次性生成完整视频）。**因此需要创建一个适配层**，拆解 Cosmos3 pipeline 的内部步骤，暴露与 WanI2V 类似的渐进去噪 API。

---

## 改动方案

### 文件变更总览

```
geo_reward/
├── __init__.py           # [修改] 新增导出
├── cosmos3_adapter.py    # [新增] Cosmos3 渐进去噪适配器
├── guidance.py           # [新增] 梯度引导模块
├── bon_pipeline.py       # [修改] 新增渐进淘汰 / 树分支 / 引导 BoN 类
├── recon_reward.py       # [修改] 新增 compute_differentiable_loss 方法
├── run_bon_cosmos3.py    # [修改] 新增 CLI 参数和模式选择
├── run_batch_bon.py      # [修改] 支持新 BoN 模式的批量运行
└── utils.py              # 不变
```

---

### 1. 新增 `geo_reward/cosmos3_adapter.py` — Cosmos3 渐进去噪适配器

**目的**：将 Cosmos3OmniPipeline 的内部步骤拆解暴露，提供与 WanI2V 对等的渐进去噪 API。

**核心类**：`Cosmos3ProgressiveAdapter`

```python
class Cosmos3ProgressiveAdapter:
    """
    Wraps Cosmos3OmniPipeline to expose step-by-step denoising control.
    
    This adapter replaces the pipeline's monolithic __call__ with fine-grained
    methods for progressive BoN, tree branching, and gradient guidance.
    """
    
    def __init__(self, pipe):
        """
        Args:
            pipe: Cosmos3OmniPipeline instance (already loaded).
        """
        self.pipe = pipe
        self.transformer = pipe.transformer  # DiT model
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.text_encoder = pipe.text_encoder
```

**需要实现的方法**（对标 WanI2V）：

#### 1.1 `prepare_progressive(prompt, image, seeds, num_frames, ...)`
- 编码文本 prompt（通过 pipeline 内部的 encode_prompt）
- 编码输入图片（通过 VAE encoder 或 pipeline 的 image 编码逻辑）
- 为每个 seed 创建**独立的 scheduler 实例**（`copy.deepcopy(self.scheduler)`）+ 独立初始噪声 latent
- 在每个 scheduler 上独立调用 `set_timesteps()`

> **关键设计**：UniPCMultistepScheduler 是有状态的（维护历史 model output 用于多步预测）。
> 必须为每个候选分配独立的 scheduler 实例，否则候选之间的历史状态会互相干扰。
> 分支操作同样需要 `copy.deepcopy(scheduler)` 来保留完整的多步历史。

- 返回 state dict：
  ```python
  {
      "candidates": [
          {
              "latent": z_i,
              "scheduler": scheduler_i,      # 每候选独立 scheduler!
              "generator": generator_i,
              "seed": seed_i,
              "step_index": 0,
          }
          for i in range(N)
      ],
      "timesteps": timesteps,               # 引用，仅用于查询长度
      "prompt_embeds": ...,
      "negative_prompt_embeds": ...,         # for CFG
      "image_latents": ...,                  # encoded first frame
      "added_cond_kwargs": ...,
      "num_frames": ...,
      "cfg_scale": ...,                      # CFG scale（注意：不叫 guidance_scale）
  }
  ```

#### 1.2 `denoise_candidates(state, alive_indices, start_step, end_step)`
- 对 alive 候选执行 `[start_step, end_step)` 的去噪步
- 每步：
  1. 取当前 timestep，从**该候选自己的 scheduler** 获取 sigma
  2. 对每个候选的 latent 运行 transformer forward（含 CFG）
  3. 调用**该候选自己的 scheduler.step()** 更新 latent（这会更新 scheduler 内部的多步历史状态）
- 返回 `(last_model_outputs, pre_step_latents)` — 最后一步的模型输出和 step 前的 latent，用于 `extract_pred_x0`

#### 1.3 `denoise_candidates_with_guidance(state, alive_indices, start_step, end_step, guidance, ...)`
- 同 `denoise_candidates`，但在每步 transformer forward 后、scheduler.step 前：
  - 检查 `guidance.should_guide(sigma_t, step_idx)`
  - 如果需要引导，调用 `guidance.guided_v_pred(latent, model_output, sigma_t, step_idx)` 修改 model_output
  - 用修改后的 model_output 执行该候选的 scheduler.step

#### 1.4 `extract_pred_x0(state, cand_idx, model_output, pre_step_latent)`
- 从 flow matching 公式计算预测的干净 latent

> **高风险点**：Cosmos3 的 prediction type 和 latent scaling 可能与 Wan2.2 不同。
> Wan2.2 使用 `x0 = x_t - sigma_t * v_pred`（标准 flow matching velocity parameterization）。
> Cosmos3 的 `UniPCMultistepScheduler` 支持多种 `prediction_type`（"epsilon", "v_prediction", "flow_matching"），
> 需要**读取 scheduler.config.prediction_type** 来确定正确的公式：
> - `"flow_matching"` / `"v_prediction"`: `x0 = x_t - sigma_t * v_pred` ✓
> - `"epsilon"`: `x0 = (x_t - sigma_t * epsilon_pred) / alpha_t`
> - 其他: 查阅 Diffusers 源码中 scheduler 的 `convert_model_output` 方法
>
> **实施时必须先打印 `scheduler.config` 确认**，不能盲目照搬 Wan2.2 公式。

#### 1.5 `find_step_for_sigma(state, target_sigma)`
- 从**任一候选的 scheduler** 读取 sigma 调度表
- 找到第一个 `sigma <= target_sigma` 的步骤索引
- 返回 `step_idx + 1` 作为 exclusive end_step

> **注意**：Diffusers UniPCMultistepScheduler 的 sigma 存储位置可能与 Wan2.2 自定义 solver 不同。
> 需要确认是通过 `scheduler.sigmas`、`scheduler.timesteps`、还是需要从 timestep 反算 sigma。

#### 1.6 `branch_candidates(state, trunk_indices, branches_per_trunk, eta, branch_seeds)`
- 从 K 个 trunk latent 创建 K × branches_per_trunk 个分支候选
- **关键**：每个分支的 scheduler 必须 `copy.deepcopy(trunk_scheduler)`，保留完整的多步历史状态
- 分支公式：`z_branch = sqrt(1 - eta²) * z_trunk + eta * sigma_t * epsilon`
  - sigma_t 从 trunk 的 scheduler 当前 step 获取
  - epsilon 使用 branch_seed 生成

> **同上高风险点**：分支公式假设的 latent space 归一化方式需要与 Cosmos3 的 scheduler 一致。
> 如果 Cosmos3 使用了 latent scaling（如 `init_noise_sigma` != 1），需要调整公式。

#### 1.7 `decode_latent(latent)`
- 使用 pipeline 的 VAE 解码 latent → 视频帧
- 返回 PIL Image 列表（通过 `cosmos3_output_to_pil` 或直接转换）

#### 1.8 `cleanup_progressive(state)`
- 释放 state 中所有 CUDA tensor（latent、scheduler、generator）
- 清空 CUDA cache

**实现要点**：
- Cosmos3 使用 flow matching，scheduler 是 `UniPCMultistepScheduler`。需要从 pipeline 源码中提取 prompt encoding、latent preparation、single-step denoising 的逻辑
- 关键是理解 Diffusers 的 `Cosmos3OmniPipeline.__call__` 内部结构，将其拆解为可控步骤。需要阅读 diffusers 库中该 pipeline 的源码来确认具体的 API 调用方式
- Transformer 的 forward 签名需要从 pipeline 源码确认

---

### 2. 修改 `geo_reward/recon_reward.py` — 新增可微分损失

从 `now` 项目移植 `compute_differentiable_loss` 及其三个子函数。这些函数用于梯度引导的反向传播。

**新增方法**：

```python
class ReconstructionReward:
    # ... existing methods ...

    def compute_differentiable_loss(self, raw_output, valid_mask, dynamic_mask, scene_scale):
        """
        Compute differentiable geometric loss for gradient guidance.
        
        loss = L_reproj + 0.5 * L_smooth + 0.3 * L_anchor
        """

    def _differentiable_reproj_loss(self, pts, extrinsics, intrinsics,
                                     static_mask, valid_mask, scene_scale):
        """Differentiable reprojection loss on first 3 consecutive pairs."""

    def _differentiable_track_smoothness(self, track, dynamic_mask, valid_mask, scene_scale):
        """Track acceleration norm on dynamic pixels."""

    def _differentiable_anchor_loss(self, pts, static_mask, extrinsic_frame0=None):
        """Depth consistency of first frame (log-space deviation from median)."""
```

同时在 `ReconRewardConfig` 中新增引导相关参数：
```python
@dataclass
class ReconRewardConfig:
    # ... existing fields ...
    
    # Gradient guidance parameters（注意：这里命名为 geo_guidance_* 避免与 CFG 冲突）
    geo_guidance_scale: float = 0.001
    geo_guidance_frequency: int = 5
    sigma_min: float = 0.08
    sigma_max: float = 0.83       # 默认 0.83 而非 0.90，与首轮淘汰点对齐
```

> **命名规范**：引导强度统一用 `geo_guidance_scale`（geometric guidance scale），
> 与 pipeline 的 `guidance_scale`（CFG scale = 6.0）严格区分。
> 下游 CLI 参数同样使用 `--geo_guidance_scale`，避免覆盖现有 `--guidance_scale`。

---

### 3. 新增 `geo_reward/guidance.py` — 梯度引导模块

从 `now` 项目移植 `GeometricGuidance` 类，适配 Cosmos3 的 VAE。

**核心区别**：
- `now` 项目使用 Wan2.2 的自定义 VAE（`Wan2_1_VAE`），有 `decode_differentiable()` 方法（带 gradient checkpointing 的可微分解码）
- Cosmos3 使用 Diffusers 内置的 VAE。需要确认 Cosmos3 VAE 是否支持可微分解码，如果不支持需要封装一个

**实现方案**：

```python
class GeometricGuidance:
    def __init__(self, model_4rc, vae, cfg=None, guidance_frames=8,
                 vae_device=None, fourrc_device=None, vae_device_2=None):
        """
        Args:
            model_4rc: 4RC model (gradient-enabled).
            vae: Cosmos3 VAE decoder.
            cfg: ReconRewardConfig with guidance parameters.
            guidance_frames: Number of frames decoded for guidance loss.
            vae_device: Device for VAE front half (multi-GPU).
            fourrc_device: Device for 4RC (multi-GPU).
            vae_device_2: Device for VAE back half (4-GPU mode).
        """
    
    def should_guide(self, sigma_t, step_idx) -> bool:
        """Check sigma window and frequency."""
    
    def guided_v_pred(self, latent, v_pred, sigma_t, step_idx):
        """
        Apply geometric guidance to v_pred.
        
        1. x0_hat = latent - sigma_t * v_pred  (需根据 prediction_type 调整)
        2. Detach, enable grad
        3. grad = _compute_guidance_gradient(x0_hat)
        4. WMReward normalization:
           scaling_t = 1 - sigma_t²
           norm_ratio = ||v_pred|| / (||grad|| + 1e-8)
           v_guided = v_pred + geo_guidance_scale * norm_ratio * scaling_t * grad
        """
    
    def _compute_guidance_gradient(self, x0_hat):
        """
        VAE decode → sample frames → 4RC forward → loss → grad.
        
        Key adaptation: Cosmos3 VAE decode path.
        """
    
    def _prepare_views(self, frames):
        """
        Differentiable PIL-free view preparation.
        F.interpolate + center crop → 4RC input format.
        """
```

**关于 Cosmos3 VAE 可微分解码**：
- Diffusers 的 VAE `decode()` 方法通常是在 `torch.no_grad()` 下调用的（在 pipeline 内部）
- 但 VAE 模型本身的 `forward()` / `decode()` 方法并不强制 `no_grad`
- 我们可以直接调用 `vae.decode(latent)` 并确保在 `torch.enable_grad()` 上下文中运行
- 如果 Cosmos3 VAE 的 decode 中有 `@torch.no_grad` 装饰器，则需要绕过（直接调用 decoder forward）
- **很可能需要 gradient checkpointing** 来控制显存（参考 now 项目中 Wan2.2 VAE 的 `decode_differentiable` 实现，该实现对 temporal 迭代做了分段 checkpoint）

---

### 4. 修改 `geo_reward/bon_pipeline.py` — 新增渐进淘汰 / 树分支 / 引导 BoN 类

**新增类层次**（对标 now 项目）：

```
Cosmos3GeoRewardBoN                    # [已有] 基础顺序 BoN
Cosmos3GeoRewardBoNProgressive         # [新增] 渐进淘汰 BoN 基类
  → Cosmos3GeoRewardBoNProgressiveV2   # [新增] 渐进淘汰 + 4RC 打分 + 模型 offload
      → Cosmos3GeoRewardBoNTreeBranching        # [新增] + 树分支
          → Cosmos3GeoRewardBoNTreeBranchingGuided  # [新增] 树分支 + 引导
Cosmos3GeoRewardOffline                # [已有] 离线打分
```

> **注意**：移除了独立的 `Cosmos3GeoRewardBoNProgressiveV2Guided` 类。
> 与 now 项目一致，**progressive + guidance（不含 tree_branching）组合不开放**。
> 原因是 progressive elimination 的 checkpoint offload 逻辑（DiT ↔ 4RC 交替上下 GPU）
> 与 guidance 的逐步 offload 逻辑（每个引导步需要 VAE+4RC on GPU）存在冲突，
> now 项目中 CLI 明确将该组合标记为 incompatible 并自动禁用 guidance flag。
> 梯度引导只支持以下两种模式：
> - **Tree branching + guidance**（推荐，多 GPU 常驻模式）
> - **Sequential BoN + guidance**（单候选逐个生成+引导，可作为降级方案）

#### 4.1 `Cosmos3GeoRewardBoNProgressive` — 基类

直接对标 `now` 项目的 `GeoRewardBoNProgressive`，将 `wan_i2v` 替换为 `Cosmos3ProgressiveAdapter`：

- 默认参数不变：`sigma_checkpoints=[0.83, 0.63]`, `elimination_ratio=0.5`, `min_survivors=2`, `score_epsilon=0.02`, `early_max_frames=12`
- `generate()`: 准备候选 → 调用 `_generate_prepared()` → cleanup
- `_eliminate()`: 固定比例淘汰逻辑（完全复用）
- `_generate_prepared()`: 抽象方法，子类实现

#### 4.2 `Cosmos3GeoRewardBoNProgressiveV2`

对标 `GeoRewardBoNProgressiveV2`，核心流程：

```
For each checkpoint (sorted by sigma descending):
    1. adapter.denoise_candidates(state, alive, cur_step, end_step)
    2. For each alive candidate:
       - If final: use final latent
       - Else: adapter.extract_pred_x0 for preview
       - adapter.decode_latent → PIL frames via cosmos3_output_to_pil
    3. Offload transformer, load 4RC
    4. Score each with recon_reward.compute_reward()
    5. Offload 4RC, reload transformer
    6. Eliminate bottom fraction
```

**适配点**：
- 输出转换：`cosmos3_output_to_pil()` 替代 `wan_output_to_pil()`
- 模型 offload：操作 `adapter.pipe.transformer` 而非 `wan.low_noise_model` / `wan.high_noise_model`
- Pipeline 加载方式需要调整（见下方"device_map 问题"）

#### 4.3 `Cosmos3GeoRewardBoNTreeBranching`

对标 `GeoRewardBoNTreeBranching`：

```
Phase 1 (Trunk): 只去噪 num_trunks 个候选到 branch_step
Phase 2 (Branch): adapter.branch_candidates() 克隆 + 深拷贝 scheduler + 噪声注入 → N 个候选
Phase 3 (Progressive): 从 branch_step 开始渐进淘汰
```

默认参数不变：`num_trunks=2`, `branches_per_trunk=4`, `branch_sigma=0.90`, `branch_eta=0.10`

#### 4.4 `Cosmos3GeoRewardBoNTreeBranchingGuided`

树分支 + 梯度引导的组合。

- 在 `_progressive_elimination()` 中使用 `denoise_candidates_with_guidance()`
- 多 GPU 常驻模式下不需要 guidance_offload_dit / guidance_reload_dit 回调
- 单 GPU 模式下不可用（显存不足以同时放 DiT + VAE + 4RC + 梯度图）

---

### 5. 修改 `geo_reward/run_bon_cosmos3.py` — 新增 CLI 参数

**新增命令行参数**（注意命名避免冲突）：

```python
# BoN 模式选择
parser.add_argument("--progressive", action="store_true",
                    help="Enable progressive elimination BoN.")
parser.add_argument("--tree_branching", action="store_true",
                    help="Enable tree branching BoN.")
parser.add_argument("--guidance", action="store_true",
                    help="Enable gradient guidance during denoising.")

# 渐进淘汰参数
parser.add_argument("--sigma_checkpoints", nargs="+", type=float,
                    default=[0.83, 0.63])
parser.add_argument("--elimination_ratio", type=float, default=0.5)
parser.add_argument("--min_survivors", type=int, default=2)
parser.add_argument("--score_epsilon", type=float, default=0.02)
parser.add_argument("--early_max_frames", type=int, default=12)

# 树分支参数
parser.add_argument("--num_trunks", type=int, default=2)
parser.add_argument("--branches_per_trunk", type=int, default=4)
parser.add_argument("--branch_sigma", type=float, default=0.90)
parser.add_argument("--branch_eta", type=float, default=0.10)

# 梯度引导参数（注意：用 geo_guidance_ 前缀，与 CFG 的 --guidance_scale 区分）
parser.add_argument("--geo_guidance_scale", type=float, default=0.001,
                    help="Geometric guidance strength (NOT CFG scale).")
parser.add_argument("--geo_guidance_frequency", type=int, default=5,
                    help="Apply guidance every N-th denoising step.")
parser.add_argument("--geo_guidance_sigma_min", type=float, default=0.08)
parser.add_argument("--geo_guidance_sigma_max", type=float, default=0.83,
                    help="Max sigma for guidance window. Default 0.83 = after first elimination.")
parser.add_argument("--guidance_frames", type=int, default=8)
```

> **已修复的命名冲突**：原方案中 `--guidance_scale` 同时用于 CFG（默认 6.0）和引导强度（默认 0.001），
> 语义完全不同且会互相覆盖。现在引导参数统一加 `geo_guidance_` 前缀。
> 现有 `--guidance_scale` (CFG) 保持不动。

**模式选择逻辑**（在 `run_bon()` 中）：

```python
# 验证兼容性（与 now 项目一致）
if args.guidance and args.progressive and not args.tree_branching:
    logger.warning(
        "--guidance is incompatible with progressive elimination (without --tree_branching). "
        "Gradient guidance only works with tree_branching or sequential BoN. "
        "Ignoring --guidance flag."
    )
    args.guidance = False

# tree_branching + guidance 时自动调整默认值（如未显式指定）
if args.guidance and args.tree_branching:
    if "--geo_guidance_frequency" not in sys.argv:
        args.geo_guidance_frequency = 3
    if "--geo_guidance_sigma_max" not in sys.argv:
        args.geo_guidance_sigma_max = 0.83

if args.tree_branching and args.guidance:
    bon = Cosmos3GeoRewardBoNTreeBranchingGuided(...)
elif args.tree_branching:
    bon = Cosmos3GeoRewardBoNTreeBranching(...)
elif args.progressive:
    bon = Cosmos3GeoRewardBoNProgressiveV2(...)
else:
    # 默认：基础顺序 BoN（保持向后兼容）
    bon = Cosmos3GeoRewardBoN(...)
```

### 6. 修改 `run_batch_bon.py` — 批量运行支持

在现有批量脚本中添加对新 BoN 模式的支持：
- 新增 `--progressive`, `--tree_branching`, `--guidance` 参数
- 创建对应的 BoN 实例（逻辑同 run_bon_cosmos3.py）
- 保持现有 resume 功能和 batch_summary 输出不变

### 7. 修改 `geo_reward/__init__.py` — 新增导出

```python
from .recon_reward import ReconstructionReward, ReconRewardConfig
from .bon_pipeline import (
    Cosmos3GeoRewardBoN,
    Cosmos3GeoRewardBoNProgressiveV2,
    Cosmos3GeoRewardBoNTreeBranching,
    Cosmos3GeoRewardBoNTreeBranchingGuided,
    Cosmos3GeoRewardOffline,
)
from .cosmos3_adapter import Cosmos3ProgressiveAdapter
from .guidance import GeometricGuidance
from .utils import cosmos3_output_to_pil, sample_frames
```

---

## 风险与不确定性

### 1. Cosmos3 prediction type 与 latent scaling（高风险 ⬆️ 从原方案"低风险"上调）
- `extract_pred_x0` 和 `branch_candidates` 的数学公式直接依赖于 scheduler 的 prediction type
- Wan2.2 使用 `x0 = x_t - sigma_t * v_pred`（flow matching velocity），Cosmos3 不一定相同
- `UniPCMultistepScheduler` 支持 `"epsilon"`, `"v_prediction"`, `"flow_matching"` 等多种 prediction type
- 如果 Cosmos3 使用了 `init_noise_sigma` 或额外的 latent scaling，分支公式也需要调整
- **应对**：Phase 1 实施时**首先打印 `scheduler.config` 和一次完整 `__call__` 的中间值**，确认 prediction type、sigma 参数化方式、latent scaling 因子后再编写公式。不盲目照搬 Wan2.2。

### 2. Cosmos3 Pipeline 内部结构（高风险）
- `Cosmos3OmniPipeline.__call__` 的内部实现细节（prompt 编码方式、latent 准备、transformer forward 签名）需要阅读 diffusers 源码确认
- **应对**：编写 adapter 前，先在服务器上用 `inspect` 或直接阅读 pipeline 源码确认内部 API

### 3. device_map 与手动 offload 冲突（中风险）
- 当前 `load_cosmos3_pipeline()` 使用 `device_map="cuda"` 加载 pipeline
- Diffusers 的 `device_map` 自动分配模块到设备，之后手动调用 `.to("cpu")` / `.to("cuda")` 可能与 accelerate 的 dispatch hook 冲突
- **应对**：对于需要手动 offload 的模式（progressive / tree branching），改为 `device_map=None` 并手动将模型放到指定设备。在 `load_cosmos3_pipeline` 中根据运行模式选择加载方式：
  ```python
  if args.progressive or args.tree_branching:
      pipe = Cosmos3OmniPipeline.from_pretrained(..., device_map=None)
      pipe.to("cuda")   # 手动控制
  else:
      pipe = Cosmos3OmniPipeline.from_pretrained(..., device_map="cuda")
  ```

### 4. Cosmos3 VAE 可微分解码（中风险）
- Diffusers VAE 的 `decode()` 可能在某些路径下不支持梯度
- 如果 Cosmos3 VAE 使用了特殊的 temporal 解码（类似 Wan2.2 的 causal conv），梯度 checkpointing 方式需要重新设计
- **应对**：先测试 `torch.enable_grad()` 下 `vae.decode()` 是否能正常反向传播；如果不行，需要写一个 `decode_differentiable` wrapper

### 5. 梯度引导显存需求（高风险 ⬆️）
- **now 项目的经验教训**：1/2/3-GPU 方案在引导模式下全部 OOM，只有 4-GPU 方案（DiT cuda:0, VAE decoder front cuda:1, VAE decoder back cuda:2, 4RC cuda:3）稳定运行
- Cosmos3-Nano (~35GB) 比 Wan2.2 14B 更大，显存压力可能更严重
- **应对**：
  - 梯度引导默认**只在 tree_branching + 多 GPU 模式下启用**
  - 在 CLI 中加入 GPU 数检测，单 GPU 或双 GPU 时自动禁用引导并给出警告
  - Phase 4 实施前先在服务器上测量各组件的实际显存占用
  - 考虑减少 `guidance_frames`（默认 8 → 可调低至 4）来降低显存需求

---

## 实施顺序

建议按以下顺序实施，每步可独立验证：

1. **Phase 1 — Cosmos3 适配器 + Scheduler 探查**：`cosmos3_adapter.py`
   - **首先在服务器上运行探查脚本**：打印 `scheduler.config`、`prediction_type`、`sigmas`、`init_noise_sigma` 等信息
   - 然后实现 `prepare_progressive` + `denoise_candidates` + `decode_latent` + `find_step_for_sigma` + `extract_pred_x0` + `cleanup_progressive`
   - 验证：单个候选生成的视频质量与原 pipeline `__call__` 一致（逐帧比较）

2. **Phase 2 — 渐进淘汰 BoN**：修改 `bon_pipeline.py`
   - 实现 `Cosmos3GeoRewardBoNProgressive` + `Cosmos3GeoRewardBoNProgressiveV2`
   - 验证：渐进淘汰确实减少了计算量并选出了高分候选

3. **Phase 3 — 树分支 BoN**：`branch_candidates` + `Cosmos3GeoRewardBoNTreeBranching`
   - 验证：分支候选质量和多样性

4. **Phase 4 — 梯度引导**：`recon_reward.py` 新增可微分损失 + `guidance.py` + `Cosmos3GeoRewardBoNTreeBranchingGuided`
   - **先在服务器上测试 Cosmos3 VAE 可微分解码可行性**
   - 这是最复杂的部分，依赖 4-GPU 环境
   - 验证：引导后的视频几何一致性分数提升

5. **Phase 5 — CLI 和集成**：`run_bon_cosmos3.py`、`run_batch_bon.py`、`__init__.py` 更新
   - 添加兼容性检查（progressive+guidance 互斥、GPU 数检测）
   - 验证：所有模式的 CLI 调用正常工作

---

## 测试验证要点

每个 Phase 需要验证：

1. **功能正确性**：生成的视频质量不低于基础 BoN
2. **数值一致性**：GeoReward 分数计算逻辑不变
3. **数学公式正确**：`extract_pred_x0` 和 `branch_candidates` 的公式适配 Cosmos3 scheduler
4. **算力节省**：渐进淘汰和树分支确实减少了 DiT forward 次数
5. **显存安全**：不会 OOM（特别是梯度引导阶段，要求 4-GPU）
6. **向后兼容**：现有基础 BoN 功能不受影响
