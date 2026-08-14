# GeoReward + BoN 集成方案：Cosmos3 I2V

## 1. 背景与目标

### 1.1 当前项目结构

```
d:\Projects\now_cosmos3\
├── cosmos-main/          # Cosmos3 官方仓库 (文档/评估/cookbooks)
│   └── cosmos-main/
└── 4RC-main/             # 4RC 4D重建模型 (已有)
    └── 4RC-main/
        ├── arc/          # 4RC 核心模型代码
        └── inference.py  # CLI 推理脚本
```

### 1.2 参考项目 (D:\Projects\now)

now 项目中已实现 Wan2.2 I2V + 4RC GeoReward + BoN 的完整流程：
- `geo_reward/recon_reward.py` — V2 重建质量奖励 (基于 4RC)
- `geo_reward/fourrc_adapter.py` — 4RC 模型接口适配器
- `geo_reward/bon_pipeline.py` — BoN 选择管线 (含渐进淘汰)
- `geo_reward/utils.py` — 帧转换工具

### 1.3 目标

将 GeoReward V2 (4RC) 的**奖励计算**和**BoN 选择机制**移植到 Cosmos3 I2V 生成流程中：
- 生成 N 个候选视频
- 用 4RC GeoReward 为每个候选打分
- 选择几何一致性最高的视频

**暂不包含**：梯度引导机制、BoN 优化策略（渐进淘汰、树分支）。

---

## 2. 关键差异分析

| 维度 | now 项目 (Wan2.2) | 本项目 (Cosmos3) |
|------|-------------------|-----------------|
| 推理框架 | 自定义 WanI2V 类 | HuggingFace Diffusers (`Cosmos3OmniPipeline`) |
| 模型规模 | 14B DiT (双模型) | 4B/16B/64B MoT |
| 调用方式 | `wan_i2v.generate(...)` | `pipe(prompt=..., image=...)` |
| 输出格式 | Tensor (3, T, H, W), range [-1,1] | `result.video` (PIL Images list / Tensor) |
| 视频尺寸 | 480×832, 81帧 | 480p/720p, 可变帧数 (默认189帧) |
| scheduler | UniPC/DPM++ Flow Matching | UniPCMultistepScheduler |
| seed 控制 | `seed` 参数 | `torch.Generator` |

### 2.1 输出格式差异

**Wan2.2 输出**：`Tensor(3, T, H, W)`, 值域 `[-1, 1]`

**Cosmos3 Diffusers 输出**：`result.video` 是一个 list of PIL Images（每帧一个 PIL Image），或可通过 `output_type="pt"` 获取 Tensor。

适配策略：编写 `cosmos3_output_to_pil()` 工具函数，统一转换为 PIL Images 列表供 GeoReward 使用。

---

## 3. 整体架构设计

```
d:\Projects\now_cosmos3\
├── cosmos-main/                   # 不修改
├── 4RC-main/                      # 不修改
└── geo_reward/                    # 新增模块 (从 now 项目移植)
    ├── __init__.py                # 模块导出
    ├── __main__.py                # python -m geo_reward 入口 (转发到 run_bon_cosmos3)
    ├── recon_reward.py            # ReconstructionReward (V2, 4RC)
    ├── fourrc_adapter.py          # 4RC 接口适配器
    ├── bon_pipeline.py            # Cosmos3 BoN 管线 (基础顺序版)
    ├── utils.py                   # 帧转换工具 (适配 Cosmos3 输出)
    └── run_bon_cosmos3.py         # CLI 入口脚本 (python -m geo_reward.run_bon_cosmos3)
```

---

## 4. 详细改动清单

### 4.1 新增 `geo_reward/` 模块

#### 4.1.1 `geo_reward/recon_reward.py`

**来源**：直接复用 `D:\Projects\now\geo_reward\recon_reward.py`

**改动**：无。此文件与视频生成器无关，仅接收 PIL Images 列表并输出分数。

核心接口：
```python
class ReconstructionReward:
    def __init__(self, model=None, device="cuda", cfg=None)
    def compute_reward(self, frames_pil, model=None) -> dict
        # Returns: {total, R_static, R_dynamic, R_motion, G_anchor, ...}
```

#### 4.1.2 `geo_reward/fourrc_adapter.py`

**来源**：直接复用 `D:\Projects\now\geo_reward\fourrc_adapter.py`

**改动**：修改 `_ensure_4rc_importable()` 中的路径搜索，保留多候选路径回退机制：
```python
candidates = [
    os.path.join(os.path.dirname(__file__), '..', '4RC-main', '4RC-main'),
    os.path.join(os.path.dirname(__file__), '..', '4RC-main'),
    os.path.join(os.path.dirname(__file__), '..', '..', '4RC-main', '4RC-main'),
]
```

#### 4.1.3 `geo_reward/utils.py`

**来源**：基于 `D:\Projects\now\geo_reward\utils.py` 改写

**改动**：
- `sample_frames()` 增加 `max_frames <= 1` 的防御（避免除零）
- `transform_to_camera()` 保持不变
- 新增 `cosmos3_output_to_pil()`，覆盖 Diffusers 可能的多种输出形态

```python
def sample_frames(total_frames=81, max_frames=20):
    """
    Uniformly sample frame indices, always including the first frame.
    """
    if max_frames <= 1:
        return [0]
    if max_frames >= total_frames:
        return list(range(total_frames))

    indices = [0]
    step = (total_frames - 1) / (max_frames - 1)
    for i in range(1, max_frames):
        indices.append(int(round(i * step)))
    return sorted(set(indices))


def cosmos3_output_to_pil(result):
    """
    Convert Cosmos3 Diffusers pipeline output to list of PIL Images.

    Handles multiple possible output formats:
    - Pipeline result object with .video attribute
    - List of PIL Images (default output_type="pil")
    - List of numpy arrays
    - Tensor (output_type="pt"): (B, T, C, H, W) / (T, C, H, W) / (B, C, T, H, W)
    - Nested list (batch wrapper): [[PIL, PIL, ...]]
    """
    import numpy as np
    from PIL import Image

    # Unwrap pipeline result object
    if hasattr(result, 'video'):
        frames = result.video
    elif hasattr(result, 'frames'):
        frames = result.frames
    elif hasattr(result, 'images'):
        frames = result.images
    else:
        frames = result

    # Unwrap batch wrapper: [[frame0, frame1, ...]] -> [frame0, frame1, ...]
    if isinstance(frames, (list, tuple)) and len(frames) == 1:
        if isinstance(frames[0], (list, tuple)):
            frames = frames[0]

    # List of PIL Images
    if isinstance(frames, (list, tuple)) and len(frames) > 0:
        if isinstance(frames[0], Image.Image):
            return list(frames)
        # List of numpy arrays
        if isinstance(frames[0], np.ndarray):
            return [Image.fromarray(f) if f.dtype == np.uint8
                    else Image.fromarray((f * 255).astype(np.uint8))
                    for f in frames]

    # Tensor path
    if isinstance(frames, torch.Tensor):
        if frames.ndim == 5:
            frames = frames[0]  # remove batch dim -> (T, C, H, W)
        if frames.ndim == 4:
            # Detect layout: (T, C, H, W) vs (C, T, H, W)
            if frames.shape[1] in (1, 3, 4) and frames.shape[0] > 4:
                pass  # already (T, C, H, W)
            elif frames.shape[0] in (1, 3, 4) and frames.shape[1] > 4:
                frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        # Normalize to [0, 255] uint8
        if frames.is_floating_point():
            if frames.min() < -0.5:
                frames = (frames + 1) / 2  # [-1, 1] -> [0, 1]
            frames = frames.clamp(0, 1)
            frames = (frames * 255).byte()
        frames = frames.cpu().numpy()
        if frames.ndim == 4:
            frames = frames.transpose(0, 2, 3, 1)  # (T, H, W, C)
        return [Image.fromarray(f) for f in frames]

    # numpy array (T, H, W, C)
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [Image.fromarray(f) if frames.dtype == np.uint8
                    else Image.fromarray((f * 255).astype(np.uint8))
                    for f in frames]

    raise ValueError(
        f"Unsupported Cosmos3 output format: type={type(frames)}, "
        f"shape={getattr(frames, 'shape', None)}"
    )
```

#### 4.1.4 `geo_reward/bon_pipeline.py`

**来源**：简化版，基于 `D:\Projects\now\geo_reward\bon_pipeline.py` 中的 `GeoRewardBoN` 逻辑

**改动**：移除渐进淘汰、树分支逻辑，仅保留基础顺序 BoN 流程。内存管理通过生成-评分交替 offload 实现（而非独立的"offloading 模块"）。

```python
class Cosmos3GeoRewardBoN:
    """
    Best-of-N pipeline for Cosmos3 I2V with GeoReward selection.
    
    Workflow:
      1. Generate N candidate videos with different seeds
      2. For each candidate, sample keyframes and compute GeoReward
      3. Select the candidate with highest total reward
    
    Memory management:
      When offload=True, after each video generation completes, the generated
      frames are saved to disk immediately and only the best video is kept in
      memory. The 4RC model is loaded to GPU only during scoring phases.
    """
    
    def __init__(self, pipe, recon_reward, max_frames=20, offload=False):
        self.pipe = pipe           # Cosmos3OmniPipeline instance
        self.reward = recon_reward # ReconstructionReward instance
        self.max_frames = max_frames
        self.offload = offload     # If True, swap Cosmos3/4RC between GPU/CPU

    def generate(self, prompt, image, N=8, num_frames=189, fps=24,
                 seed_base=None, save_all=False, output_dir=None,
                 **pipe_kwargs) -> (best_video, all_rewards, best_index):
        """
        Generate N candidates and select best by GeoReward.
        
        Args:
            save_all: If True, save all candidate videos to output_dir.
                      If False (default), only keep the best in memory.
            output_dir: Directory for saving videos (required if save_all=True).
        
        Returns:
            (best_video_frames, all_rewards_list, best_index)
        """
        ...
```

同时提供离线评分类：

```python
class Cosmos3GeoRewardOffline:
    """Score pre-generated videos (PIL frame lists or video files)."""
    
    def __init__(self, recon_reward, max_frames=20):
        ...
    
    def score_videos(self, video_paths_or_frames) -> list:
        ...
    
    def select_best(self, video_paths_or_frames) -> (int, list):
        ...
```

#### 4.1.5 `geo_reward/__init__.py`

```python
from .recon_reward import ReconstructionReward, ReconRewardConfig
from .bon_pipeline import Cosmos3GeoRewardBoN, Cosmos3GeoRewardOffline
from .utils import cosmos3_output_to_pil, sample_frames
```

#### 4.1.6 `geo_reward/run_bon_cosmos3.py`

CLI 入口脚本，类似 now 项目的 `run_bon_v2.py`，但适配 Cosmos3。
使用 `python -m geo_reward.run_bon_cosmos3` 调用，避免相对导入问题。

```python
"""
Cosmos3 I2V + GeoReward Best-of-N Pipeline.

Usage:
    python -m geo_reward.run_bon_cosmos3 \
        --model nvidia/Cosmos3-Nano \
        --fourrc_model Luo-Yihang/4RC \
        --image /path/to/first_frame.png \
        --prompt "robot picks up the red block" \
        --N 8 --num_frames 189 --fps 24
"""
```

主要参数：
- `--model`: Cosmos3 HuggingFace 模型 ID (nvidia/Cosmos3-Nano 或 nvidia/Cosmos3-Super)
- `--fourrc_model`: 4RC 模型路径，接受 HuggingFace repo ID (如 "Luo-Yihang/4RC") 或本地 checkpoint 目录
- `--image`: 输入首帧图像
- `--prompt`: 文本提示
- `--N`: 候选视频数量
- `--num_frames`: 输出帧数 (默认 189)
- `--fps`: 帧率 (默认 24)
- `--height / --width`: 分辨率 (默认 720/1280)
- `--num_inference_steps`: 去噪步数 (默认 35)
- `--guidance_scale`: CFG scale (默认 6.0)
- `--seed_base`: 基础随机种子
- `--output_dir`: 输出目录
- `--save_all`: 是否保存所有候选视频到磁盘 (默认只保存 best)
- `--offload`: 启用 Cosmos3/4RC 交替 offload (显存不足时使用)
- `--max_frames`: GeoReward 采样帧数 (默认 20)
- `--image_size`: 4RC 输入分辨率 (默认 518)
- Reward 权重参数 (static/dynamic/motion weight 等)

---

## 5. 执行流程

### 5.1 基础 BoN 流程

```
输入：首帧图像 + 文本提示 + N (候选数)
       │
       ▼
┌─────────────────────────────────────┐
│  1. 加载 Cosmos3 Pipeline (Diffusers)  │
│  2. 加载 4RC 模型                      │
│  3. 初始化 ReconstructionReward        │
└─────────────────────────────────────┘
       │
       ▼
┌──────────────── Loop i = 0..N-1 ────────────────┐
│                                                   │
│  seed = seed_base + i                            │
│  generator = torch.Generator("cuda").manual_seed(seed)  │
│                                                   │
│  [offload=True: 确保 Cosmos3 在 GPU, 4RC 在 CPU]  │
│                                                   │
│  result = pipe(                                   │
│      prompt=prompt,                               │
│      image=image,                                 │
│      num_frames=num_frames,                       │
│      fps=fps,                                     │
│      generator=generator,                         │
│      **pipe_kwargs                                │
│  )                                                │
│                                                   │
│  frames_pil = cosmos3_output_to_pil(result)      │
│  indices = sample_frames(len(frames_pil), max_frames) │
│  sampled = [frames_pil[i] for i in indices]       │
│                                                   │
│  [offload=True: Cosmos3 → CPU, 4RC → GPU]        │
│                                                   │
│  reward_dict = recon_reward.compute_reward(sampled)│
│                                                   │
│  if save_all: 保存当前视频到磁盘                   │
│  else: 仅保留 best_so_far 在内存，释放当前帧       │
│                                                   │
│  rewards.append(reward_dict)                      │
└───────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  best_idx = argmax(rewards, key=total) │
│  保存最佳视频 + 奖励日志              │
└─────────────────────────────────────┘
```

### 5.2 内存管理策略

由于 Cosmos3 模型 (16B~64B) + 4RC 可能无法同时放在 GPU 上，
BoN pipeline 通过 `offload=True` 参数启用交替 offload：

1. **offload=False (默认，足够显存)**：Cosmos3 和 4RC 同时在 GPU，无需搬运
2. **offload=True (显存不足)**：
   - 生成阶段：Cosmos3 在 GPU，4RC 在 CPU
   - 评分阶段：Cosmos3 通过 `pipe.to("cpu")` 卸载，4RC `model.cuda()` 加载到 GPU
   - 每个候选视频生成后立即评分，然后交替回去

### 5.3 候选视频存储策略

N=8 且 189 帧时，保留所有候选完整视频在内存中压力很大。策略：

- **默认 (save_all=False)**：只在内存中保留当前最佳视频帧，每个候选评分后释放
- **save_all=True**：每个候选生成后立即写入磁盘 (output_dir)，内存中仍只保留 best
- 最终结果：内存中只有 1 份最佳视频 + N 份 reward dict (很小)

---

## 6. 文件依赖关系

```
geo_reward/run_bon_cosmos3.py
    ├── diffusers.Cosmos3OmniPipeline    (pip install)
    ├── geo_reward/bon_pipeline.py
    │   ├── geo_reward/recon_reward.py
    │   │   └── geo_reward/fourrc_adapter.py
    │   │       └── 4RC-main/4RC-main/arc/  (sys.path)
    │   └── geo_reward/utils.py
    └── PIL, torch, numpy, etc.
```

---

## 7. 预期使用方式

### 7.1 命令行使用

```bash
# 基础 BoN (8 候选)
python -m geo_reward.run_bon_cosmos3 \
    --model nvidia/Cosmos3-Nano \
    --fourrc_model Luo-Yihang/4RC \
    --image examples/robot_scene.png \
    --prompt "The robot arm reaches for and grasps the red cube on the table" \
    --N 8 \
    --num_frames 189 \
    --fps 24 \
    --height 480 --width 832 \
    --output_dir outputs/bon_cosmos3

# 离线评分
python -m geo_reward.run_bon_cosmos3 \
    --mode score \
    --fourrc_model ./4RC-main/4RC-main \
    --video_dir outputs/generated_videos/

# 显存不足时启用 offload + 只保存最佳视频
python -m geo_reward.run_bon_cosmos3 \
    --model nvidia/Cosmos3-Nano \
    --fourrc_model Luo-Yihang/4RC \
    --image examples/robot_scene.png \
    --prompt "robot picks up the red block" \
    --N 8 --offload \
    --output_dir outputs/bon_cosmos3
```

### 7.2 Python API 使用

```python
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from geo_reward import Cosmos3GeoRewardBoN, ReconstructionReward, ReconRewardConfig
from arc.models.arc.arc import Arc

# 加载 Cosmos3
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0)

# 加载 4RC (支持 HF repo ID 或本地路径)
fourrc_model = Arc.from_pretrained("Luo-Yihang/4RC")  # 或 "./4RC-main/4RC-main"
fourrc_model = fourrc_model.cuda().eval()

# 初始化 GeoReward
reward = ReconstructionReward(model=fourrc_model, device="cuda")

# 创建 BoN pipeline
bon = Cosmos3GeoRewardBoN(pipe=pipe, recon_reward=reward, max_frames=20)

# 运行
from PIL import Image
img = Image.open("first_frame.png")
best_video, rewards, best_idx = bon.generate(
    prompt="robot picks up the red block",
    image=img,
    N=8,
    num_frames=189,
    fps=24,
    height=480,
    width=832,
    guidance_scale=6.0,
    num_inference_steps=35,
)
# best_video: list[PIL.Image] (仅最佳视频帧)
# rewards: list[dict] (所有 N 个候选的 reward)
# best_idx: int
```

---

## 8. 验证计划

1. **单元测试**：确认 `ReconstructionReward.compute_reward()` 在给定 PIL 帧列表时正确返回分数
2. **集成测试**：使用 Cosmos3-Nano (4B) 生成 2 个候选视频，验证 BoN 选择流程端到端正常工作
3. **对比验证**：对同一输入图像，比较 BoN 选出的视频 vs 随机种子视频的 GeoReward 分数

---

## 9. 注意事项

1. **Diffusers 版本**：需要最新 diffusers (支持 `Cosmos3OmniPipeline`)，通过 `pip install git+https://github.com/huggingface/diffusers.git` 安装
2. **4RC 导入**：通过 sys.path 加入 `4RC-main/4RC-main/` 目录使 `import arc` 可用
3. **显存**：Cosmos3-Nano (4B) 约需 16GB，4RC 约需 4-6GB。如果总显存不足，启用 offload 模式
4. **帧数适配**：Cosmos3 默认 189 帧，GeoReward 默认采样 20 帧用于评分
5. **分辨率适配**：4RC 输入为 518px（长边），会自动 resize，不需要生成端做特殊处理
6. **不修改任何现有代码**：所有改动均为新增文件

---

## 10. 后续扩展 (本次不实现)

- **渐进淘汰 BoN**：需要 hook 进 Cosmos3 的 denoising loop，获取中间 latent 并提前淘汰低分候选
- **梯度引导**：需要 differentiable VAE decode + 4RC forward，在 denoising step 中注入几何梯度
- **树分支**：需要在 denoising 中间点进行 latent 分支 (噪声注入)
- **BoN 优化策略**：自适应 N、早停等

这些扩展都依赖对 Diffusers scheduler/pipeline 内部 loop 的深度访问，需要后续研究 `Cosmos3OmniPipeline` 的源码来确定可行方案。
