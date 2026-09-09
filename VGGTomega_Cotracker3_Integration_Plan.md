# VGGTomega + CoTracker3 集成方案文档

## 0. 审查意见修正总结

本方案已根据审查意见进行全面修正，所有 P0 和 P1 问题均已解决：

### ✅ P0 必须修复项（已完成）

1. **不重复定义 `cosmos3_output_to_pil`**
   - ❌ 原方案：在 `geo_reward/utils.py` 末尾添加简化版
   - ✅ 修正后：直接复用现有的完整版本（`geo_reward/utils.py:29`），不做任何修改
   - ✅ 新增独立的 `VGGTomega_Cotracker3/utils.py` 提供 `save_video_from_pil`

2. **`flow_shift` 设置方式**
   - ❌ 原方案：在每次 `pipe()` 调用时透传 `flow_shift=args.flow_shift`
   - ✅ 修正后：在 scheduler 初始化时设置
   ```python
   pipe.scheduler = UniPCMultistepScheduler.from_config(
       pipe.scheduler.config, flow_shift=args.flow_shift
   )
   ```
   - 参考：`geo_reward/run_bon_cosmos3.py:220`、`run_batch_bon.py:176`

3. **统一使用 `cosmos3_output_to_pil`**
   - ❌ 原方案：硬编码 `output.frames[0]`
   - ✅ 修正后：`frames_pil = cosmos3_output_to_pil(result)`
   - 位置：`bon_pipeline_vggt.py` 的 `generate` 方法

4. **使用 `torch.bfloat16`**
   - ❌ 原方案：`torch_dtype=torch.float16`
   - ✅ 修正后：`torch_dtype=torch.bfloat16`（与现有代码一致）
   - 参考：`geo_reward/run_bon_cosmos3.py:210`、`run_batch_bon.py:163`

### ✅ P1 建议补齐项（已完成）

1. **生成器设备兼容性**
   - ✅ 添加设备检测逻辑（兼容 meta/cpu/cuda）
   ```python
   pipe_device = getattr(self.pipe, "device", None)
   if pipe_device is None or str(pipe_device) == "meta":
       gen_device = "cuda" if torch.cuda.is_available() else "cpu"
   else:
       gen_device = pipe_device
   ```
   - 参考：`geo_reward/bon_pipeline.py:100-105`

2. **添加 `opencv-python` 依赖**
   - ✅ 在依赖清单中显式添加（第 7.1 节）
   - 理由：`save_video_from_pil` 依赖 `cv2`

3. **清理旧包名**
   - ✅ 方案中统一使用 `VGGTomega_Cotracker3`
   - ⚠️ 实施时需清理已复制文件中的旧包名 `VGGT_Cotracker3`

### ✅ 核心约束（已确认）

- ✅ **不改动 `geo_reward/` 和 `4RC-main/`**
  - 所有修改仅限于 `VGGTomega_Cotracker3/` 目录
  - 复用 `geo_reward/utils.py` 的现有函数，不做任何修改
  - 保证现有 4RC 流程完全不受影响

---

### 1.1 当前项目状态
- **now_cosmos3 项目**：基于 Cosmos3 I2V 模型 + 4RC 重建模型的视频生成与几何评分系统
- **now 项目**：基于 Wan2.2 I2V 模型 + 4RC 重建模型，已验证 VGGTomega + CoTracker3 作为替代重建后端的可行性

### 1.2 集成目标
将 now 项目中验证成功的 **VGGTomega + CoTracker3** 重建方法移植到 now_cosmos3 项目，实现：
1. 使用 VGGTomega 获取 depth、camera pose、intrinsic
2. 使用 CoTracker3 获取 dense tracking（逐像素轨迹追踪）
3. 组合两者输出，生成与 4RC 兼容的几何数据格式
4. 使用相同的 GeoReward 公式评分（验证不同重建模型的通用性）

### 1.3 核心价值
- **验证通用性**：证明 GeoReward 方法不依赖特定重建模型（4RC → VGGTomega+CoTracker3）
- **模型对比**：对比不同重建后端的评分结果和计算效率
- **技术储备**：为未来集成更多重建模型（如 DepthAnything、MoGe 等）提供架构参考

---

## 2. 技术架构

### 2.1 目录结构
```
now_cosmos3/
├── VGGTomega/                          # [已复制] VGGTomega 模型代码
│   └── vggt-omega-main/
├── co-tracker-main/                    # [已复制] CoTracker3 模型代码
│   └── co-tracker-main/
├── VGGTomega_Cotracker3/               # [已复制] 核心集成代码
│   ├── __init__.py
│   ├── vggt_omega_adapter.py          # VGGTomega 适配器
│   ├── cotracker3_adapter.py          # CoTracker3 适配器
│   ├── combo_adapter.py               # 组合适配器（VGGTomega + CoTracker3）
│   ├── recon_reward_vggt.py           # VGGT 后端的 GeoReward 计算
│   ├── bon_pipeline_vggt.py           # VGGT 后端的 Best-of-N pipeline
│   ├── run_bon_vggt.py                # 单条测试 CLI
│   └── run_bon_batch_vggt.py          # 批量测试 CLI
├── geo_reward/                         # [现有] Cosmos3 + 4RC 的 GeoReward
│   ├── bon_pipeline.py
│   ├── cosmos3_adapter.py
│   ├── fourrc_adapter.py
│   ├── recon_reward.py
│   └── ...
└── cosmos-main/                        # [现有] Cosmos3 模型代码
```

### 2.2 模块依赖关系
```
┌─────────────────────────────────────────────────────────┐
│                 Cosmos3 I2V 视频生成                     │
│            (cosmos-main/Cosmos3OmniPipeline)            │
└─────────────────────────────────────────────────────────┘
                          ↓ 生成视频
        ┌─────────────────────────────────────┐
        │                                     │
        ↓                                     ↓
┌───────────────────┐            ┌────────────────────────┐
│  4RC 重建后端      │            │ VGGTomega + CoTracker3  │
│  (geo_reward/)    │            │ 重建后端                 │
│                   │            │ (VGGTomega_Cotracker3/)│
│  • fourrc_adapter │            │                         │
│  • recon_reward   │            │ • vggt_omega_adapter    │
│  • bon_pipeline   │            │ • cotracker3_adapter    │
│                   │            │ • combo_adapter         │
│                   │            │ • recon_reward_vggt     │
│                   │            │ • bon_pipeline_vggt     │
└───────────────────┘            └────────────────────────┘
        │                                     │
        │   输出相同格式的几何数据              │
        │   (pts, track, extrinsic,           │
        │    intrinsic, visibility)           │
        └─────────────────┬───────────────────┘
                          ↓
              ┌──────────────────────┐
              │   GeoReward 评分      │
              │   (相同的 reward 公式) │
              └──────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 VGGTomega 适配器 (`vggt_omega_adapter.py`)

#### 功能
封装 VGGTomega 模型的加载、预处理和推理，输出标准化的几何数据。

#### 核心函数
```python
def load_vggt_omega(checkpoint_path, device="cuda") -> VGGTOmega
    """加载 VGGTomega 模型"""

def preprocess_frames(frames_pil, image_resolution=512, patch_size=16) -> Tensor
    """
    将 PIL 帧列表转为 VGGTomega 输入格式 (N, 3, H, W)
    使用 balanced 模式：保持总 token 数恒定
    """

def run_vggt_omega_inference(model, images, device="cuda") -> dict
    """
    运行 VGGTomega 推理
    
    Returns:
        dict:
            - depth: (N, H, W) 深度图
            - extrinsic_c2w: (N, 4, 4) camera-to-world 变换
            - intrinsic: (N, 3, 3) 相机内参
            - pts: (N, H, W, 3) 世界坐标 3D 点
            - resolution: (H, W) 输出分辨率
    """
```

#### 关键设计
- **坐标转换**：VGGTomega 输出 camera-from-world (3,4)，需转换为 camera-to-world (4,4)
- **分辨率处理**：balanced 模式保持 token 数 `(resolution/patch_size)^2` 恒定
- **宽高比限制**：限制到 [0.5, 2.0]，超出范围进行中心裁剪

---

### 3.2 CoTracker3 适配器 (`cotracker3_adapter.py`)

#### 功能
封装 CoTracker3 模型的加载和逐像素 dense tracking。

#### 核心函数
```python
def load_cotracker3(checkpoint_path=None, device="cuda") -> CoTrackerPredictor
    """
    加载 CoTracker3 offline 模型
    checkpoint_path=None 时使用 torch.hub 自动下载
    """

def frames_to_video_tensor(frames_pil, target_size=None) -> Tensor
    """
    将 PIL 帧列表转为 CoTracker3 输入格式 (1, T, 3, H, W)
    值域 [0, 255]
    """

def run_dense_tracking(model, video, query_frame=0, 
                       chunk_size=4096, device="cuda") -> tuple
    """
    逐像素 dense tracking：对 query_frame 的每个像素追踪到所有帧
    
    对 H*W 个像素生成 queries，分 chunk 调用 CoTracker3
    
    Returns:
        tracks_2d: (T, H, W, 2) 每个像素在每帧的 (x, y) 坐标
        visibility: (T, H, W) bool，每个像素在每帧的可见性
    """
```

#### 关键设计
- **分 chunk 处理**：H×W 像素一次性追踪会 OOM，按 chunk_size 分批（默认 4096）
- **坐标约定**：queries 格式 (frame_idx, x, y)，x=列坐标，y=行坐标
- **分辨率对齐**：将输入帧 resize 到与 VGGTomega 相同分辨率，确保坐标对齐

---

### 3.3 组合适配器 (`combo_adapter.py`)

#### 功能
组合 VGGTomega 和 CoTracker3 的输出，生成与 4RC 兼容的完整几何数据。

#### 核心流程
```python
def run_combo_inference(vggt_model, cotracker_model, frames_pil,
                        image_resolution=512, chunk_size=4096, 
                        device="cuda") -> dict
    """
    组合推理流程：
    
    步骤 1: VGGTomega 推理
        → depth, extrinsic_c2w, intrinsic, pts (N, H_v, W_v, 3)
    
    步骤 2: CoTracker3 dense tracking
        → tracks_2d (T, H_v, W_v, 2), visibility (T, H_v, W_v)
        (resize 输入帧到 VGGTomega 的分辨率 H_v×W_v)
    
    步骤 3: 2D tracks → 3D displacement
        对每帧 t 的每个像素 (h, w):
          1. 从 tracks_2d 获取该像素在帧 t 的 2D 位置 (x_t, y_t)
          2. 在帧 t 的 depth map 上双线性插值获取 depth_t
          3. 反投影 (x_t, y_t, depth_t) 到世界坐标 pts_3d_t
          4. track[t, h, w] = pts_3d_t - pts[0, h, w]
          5. 不可见像素: track 设为 0
    
    Returns:
        dict:
            - pts: (N, H, W, 3) 世界坐标 3D 点
            - track: (N, H, W, 3) 逐像素 3D 位移（相对帧 0）
            - extrinsic: (N, 4, 4) camera-to-world
            - intrinsic: (N, 3, 3) 内参
            - visibility: (N, H, W) 逐帧逐像素可见性
    """
```

#### 关键设计
- **双线性插值采样**：使用 `F.grid_sample` 在 depth map 上插值
- **坐标归一化**：grid_sample 需要 [-1, 1] 归一化坐标
- **数值稳定性**：depth clamp 到 min=0.01，替换 NaN/Inf
- **遮挡处理**：CoTracker3 visibility 标记物理遮挡（vs 4RC 的模型置信度）

---

### 3.4 GeoReward 计算 (`recon_reward_vggt.py`)

#### 功能
使用 VGGTomega + CoTracker3 后端计算 GeoReward，公式与 4RC 版本完全一致。

#### 核心类
```python
@dataclass
class VGGTReconRewardConfig:
    """
    配置类，与 ReconRewardConfig 参数完全一致
    去掉 conf_valid_quantile（4RC 专用）和梯度引导参数
    """
    # Weights
    static_weight: float = 0.50
    dynamic_weight: float = 0.30
    motion_weight: float = 0.20
    
    # Dynamic mask
    dynamic_threshold_ratio: float = 0.01
    
    # R_static
    tau_reproj: float = 0.05
    occlusion_margin: float = 1.05
    
    # R_dynamic
    tau_accel: float = 0.02
    tau_speed: float = 1.5
    max_sample_pixels: int = 1000
    
    # R_motion
    tau_cam: float = 0.02
    tau_rot: float = 0.05
    min_motion: float = 0.005
    tau_motion: float = 0.02
    
    # Frame sampling
    max_frames: int = 20
    image_size: int = 512  # VGGTomega 输入分辨率
    
    # CoTracker3
    chunk_size: int = 4096

class VGGTReconstructionReward:
    """
    VGGTomega + CoTracker3 后端的 Reconstruction Quality Reward
    
    使用与 4RC 版本完全相同的 Reward 公式，差异仅在：
    - 几何后端：4RC → VGGTomega + CoTracker3 组合
    - R_static: 去掉 valid_geo（conf-based mask），所有像素参与
    - R_dynamic: 用 CoTracker3 visibility 替代 valid_track
    - 不包含梯度引导相关功能
    """
    
    def compute_reward(self, frames_pil, vggt_model=None, 
                       cotracker_model=None) -> dict:
        """
        计算 GeoReward
        
        Returns:
            dict:
                - total: float, 最终 reward [0, 1]
                - version: "vggt_v2"
                - R_static, R_dynamic, R_motion, G_anchor: float
                - scene_scale, dynamic_ratio, visibility_ratio: float
        """
```

#### Reward 公式
```
Total = G_anchor × (0.50×R_static + 0.30×R_dynamic + 0.20×R_motion)

其中：
- G_anchor: 首帧几何合理性门控（深度有效性）
- R_static: 静态区域跨帧深度重投影一致性
- R_dynamic: 动态区域轨迹加速度 + 覆盖率 + 速度平滑度
- R_motion: 相机平滑度 + motion gate + 瞬移 penalty
```

#### 与 4RC 版本的差异
| 组件 | 4RC 版本 | VGGT 版本 | 说明 |
|------|---------|-----------|------|
| R_static mask | `static_mask & valid_geo` | `static_mask` | 去掉 conf-based 过滤 |
| R_dynamic mask | `dynamic_mask & valid_track` | `dynamic_mask & visibility` | 用物理遮挡替代置信度 |
| 梯度引导 | 支持 | 不支持 | VGGT 是预训练模型，不需要引导 |

---

### 3.5 Best-of-N Pipeline (`bon_pipeline_vggt.py`)

#### 功能
简单 Best-of-N pipeline，全量生成 N 个候选视频，统一评分，选出最优。

#### 核心类
```python
class GeoRewardBoNVGGT:
    """
    简单 Best-of-N pipeline（VGGTomega + CoTracker3 后端）
    
    流程：
    1. Cosmos3 生成 N 个候选视频（完整去噪）
    2. VAE decode 所有候选
    3. 对每个候选：均匀抽帧 → VGGT+CoTracker3 推理 → 计算 reward
    4. 选 reward 最高的候选
    
    显存管理：
    - 生成阶段: DiT + VAE 在 GPU
    - 评分阶段: VGGT + CoTracker3 在 GPU，DiT + VAE 卸载到 CPU
    """
    
    def __init__(self, cosmos3_pipe, vggt_model_path, 
                 cotracker_model_path=None, N=8, 
                 num_frames_for_reward=20, reward_config=None,
                 chunk_size=4096, device="cuda"):
        """
        Args:
            cosmos3_pipe: Cosmos3OmniPipeline 实例
            vggt_model_path: VGGTomega 权重路径
            cotracker_model_path: CoTracker3 权重路径（None 用 torch.hub）
            N: 候选数
            num_frames_for_reward: reward 评分用的帧数
            reward_config: VGGTReconRewardConfig 实例
            chunk_size: CoTracker3 per-chunk 点数
            device: 设备
        """
    
    def generate(self, image, prompt, N=None, num_frames=189,
                 fps=24, seed_base=None, output_dir=None, 
                 save_fn=None, **cosmos3_kwargs) -> dict:
        """
        执行 BoN 生成
        
        Returns:
            dict:
                - best_video: 最优视频 tensor
                - best_score: 最优分数
                - best_seed: 最优种子
                - all_scores: 所有候选分数列表
                - all_details: 所有候选的详细 reward 分解
                - seeds: 所有候选的种子列表
                - total_time_sec: 总用时
        """
```

#### 显存管理策略
```python
# 阶段 1: 生成候选（DiT + VAE 在 GPU）
for i in range(N):
    video = cosmos3_pipe.generate(...)
    candidates.append(video.cpu())  # 立即卸载到 CPU
    torch.cuda.empty_cache()

# 阶段 2: 卸载生成模型，加载评分模型
self._offload_dit()
self._offload_vae()
self._load_reward_models()  # VGGT + CoTracker3 → GPU

# 阶段 3: 评分
for video_tensor, seed in candidates:
    frames_pil = cosmos3_output_to_pil(video_tensor)
    sampled = [frames_pil[i] for i in indices]
    r = self.reward.compute_reward(sampled)

# 阶段 4: 卸载评分模型，恢复 VAE（用于后续保存）
self._offload_reward_models()
self._load_vae()
```

---

## 4. 适配 Cosmos3 所需的修改

### 4.1 当前 now 项目中的依赖
now 项目中的 VGGTomega_Cotracker3 代码依赖于 **Wan2.2** 的 I2V 模型：

```python
# now/VGGTomega_Cotracker3/bon_pipeline_vggt.py
from geo_reward.utils import wan_output_to_pil, sample_frames  # ← Wan2.2 专用
```

```python
# now/VGGTomega_Cotracker3/run_bon_vggt.py
import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
```

### 4.2 需要修改的文件

#### 4.2.1 修改 `VGGTomega_Cotracker3/__init__.py`
**目的**：导出核心类和函数

```python
"""
VGGTomega + CoTracker3 组合重建后端。

提供与 4RC 兼容的几何数据输出和 GeoReward 评分。
"""

from .vggt_omega_adapter import (
    load_vggt_omega,
    preprocess_frames,
    run_vggt_omega_inference,
)

from .cotracker3_adapter import (
    load_cotracker3,
    frames_to_video_tensor,
    run_dense_tracking,
)

from .combo_adapter import run_combo_inference

from .recon_reward_vggt import (
    VGGTReconstructionReward,
    VGGTReconRewardConfig,
)

from .bon_pipeline_vggt import GeoRewardBoNVGGT

__all__ = [
    # Adapters
    "load_vggt_omega",
    "preprocess_frames",
    "run_vggt_omega_inference",
    "load_cotracker3",
    "frames_to_video_tensor",
    "run_dense_tracking",
    "run_combo_inference",
    # Reward
    "VGGTReconstructionReward",
    "VGGTReconRewardConfig",
    # Pipeline
    "GeoRewardBoNVGGT",
]
```

#### 4.2.2 修改 `VGGTomega_Cotracker3/bon_pipeline_vggt.py`
**目的**：替换 Wan2.2 依赖为 Cosmos3

**关键修改点**：

1. **导入 utils**（复用现有 geo_reward 工具，不修改它）
```python
# 原始代码（now 项目）
from geo_reward.utils import wan_output_to_pil, sample_frames  # ← Wan2.2 专用

# 修改后（now_cosmos3 项目）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_reward.utils import cosmos3_output_to_pil, sample_frames  # ← Cosmos3 专用
```

2. **修改 `__init__` 方法**：
```python
def __init__(
    self,
    cosmos3_pipe,              # ← 改为 Cosmos3OmniPipeline
    vggt_model_path,
    cotracker_model_path=None,
    N=8,
    num_frames_for_reward=20,
    reward_config=None,
    chunk_size=4096,
    device="cuda",
):
    """
    Args:
        cosmos3_pipe: Cosmos3OmniPipeline 实例  # ← 改注释
        vggt_model_path: VGGTomega 权重路径
        cotracker_model_path: CoTracker3 权重路径（None 用 torch.hub）
        N: 候选数
        num_frames_for_reward: reward 评分用的帧数
        reward_config: VGGTReconRewardConfig 实例
        chunk_size: CoTracker3 per-chunk 点数
        device: 设备
    """
    self.pipe = cosmos3_pipe  # ← 改变量名
    self.N = N
    # ... 其余不变
```

3. **修改 `_offload_dit` 方法**：
```python
def _offload_dit(self):
    """将 DiT 卸载到 CPU。"""
    # Cosmos3 的 DiT 模型名称是 transformer
    if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
        self.pipe.transformer.cpu()
    torch.cuda.empty_cache()
```

4. **修改 `generate` 方法**（关键修复）：
```python
def generate(self, image, prompt, N=None, num_frames=189,
             fps=24, seed_base=None, output_dir=None, save_fn=None,
             **cosmos3_kwargs):  # ← 改参数名
    """
    执行 BoN 生成。

    Args:
        image: 首帧图片（PIL.Image）
        prompt: 动作指令文本
        N: 候选数（覆盖 self.N）
        num_frames: 视频帧数
        fps: 帧率
        seed_base: 基础种子
        output_dir: 输出目录（保存所有候选视频）
        save_fn: callable(frames_pil, path) 保存视频
        **cosmos3_kwargs: 传给 Cosmos3OmniPipeline 的其他参数
                         (height, width, guidance_scale, num_inference_steps)
                         ⚠️ 不要传 flow_shift（在 scheduler 上已设置）

    Returns:
        dict:
          - best_video: 最优视频 (List[PIL.Image])
          - best_score: 最优分数
          - best_seed: 最优种子
          - all_scores: 所有候选分数列表
          - all_details: 所有候选的详细 reward 分解
    """
    N = N or self.N
    if seed_base is None:
        seed_base = random.randint(0, 2**31 - 1)

    indices = sample_frames(num_frames, self.num_frames_for_reward)

    t_start = time.time()

    # ===== 阶段 1: 生成 N 个候选 =====
    print(f"\n[BoNVGGT] 开始生成 {N} 个候选视频 (seeds {seed_base}..{seed_base+N-1})")

    # 生成器设备兼容性检测（参考 geo_reward/bon_pipeline.py:100-105）
    pipe_device = getattr(self.pipe, "device", None)
    if pipe_device is None or str(pipe_device) == "meta":
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        gen_device = pipe_device

    candidates = []  # (frames_pil, seed)
    for i in range(N):
        seed = seed_base + i
        t0 = time.time()

        # Cosmos3 生成
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            fps=fps,
            generator=generator,
            **cosmos3_kwargs,
        )
        # ✅ 使用 cosmos3_output_to_pil 统一处理输出
        frames_pil = cosmos3_output_to_pil(result)
        gen_time = time.time() - t0

        if frames_pil is None or len(frames_pil) == 0:
            print(f"  候选 {i+1}/{N} (seed={seed}): 生成失败，跳过")
            continue

        candidates.append((frames_pil, seed))
        print(f"  候选 {i+1}/{N} (seed={seed}): 生成完成 [{gen_time:.1f}s]")

        torch.cuda.empty_cache()

    if not candidates:
        raise RuntimeError("所有候选生成失败。")

    # ===== 阶段 2: 卸载 DiT+VAE，加载评分模型 =====
    print(f"\n[BoNVGGT] 卸载 DiT+VAE，加载 VGGTomega + CoTracker3...")
    self._offload_dit()
    self._offload_vae()
    self._load_reward_models()

    # ===== 阶段 3: 评分 =====
    print(f"\n[BoNVGGT] 对 {len(candidates)} 个候选评分...")
    all_scores = []
    all_details = []

    for idx, (frames_pil, seed) in enumerate(candidates):
        t0 = time.time()

        sampled = [frames_pil[i] for i in indices if i < len(frames_pil)]

        r = self.reward.compute_reward(sampled)
        reward_time = time.time() - t0

        all_scores.append(r["total"])
        all_details.append(r)

        print(f"  候选 {idx+1}/{len(candidates)} (seed={seed}): "
              f"total={r['total']:.4f} "
              f"(R_static={r['R_static']:.4f}, "
              f"R_dynamic={r['R_dynamic']:.4f}, "
              f"R_motion={r['R_motion']:.4f}, "
              f"G_anchor={r['G_anchor']:.2f}) "
              f"[{reward_time:.1f}s]")

    # ===== 阶段 4: 选出最优 =====
    best_idx = max(range(len(all_scores)),
                   key=lambda i: all_scores[i] if np.isfinite(all_scores[i]) else -float("inf"))
    best_frames, best_seed = candidates[best_idx]
    best_score = all_scores[best_idx]

    elapsed = time.time() - t_start
    print(f"\n[BoNVGGT] 最优: seed_{best_seed} (total={best_score:.4f}) "
          f"总用时 {elapsed:.1f}s")

    # ===== 阶段 5: 卸载评分模型，恢复 VAE =====
    self._offload_reward_models()
    self._load_vae()

    # ===== 保存视频 =====
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        ranked_indices = sorted(range(len(all_scores)),
                                key=lambda i: all_scores[i], reverse=True)
        for rank, orig_idx in enumerate(ranked_indices):
            frames, s = candidates[orig_idx]
            reward_val = all_scores[orig_idx]
            suffix = "_BEST" if orig_idx == best_idx else ""
            filename = f"candidate_{rank+1:02d}_r{reward_val:.4f}_seed{s}{suffix}.mp4"
            path = os.path.join(output_dir, filename)
            if save_fn is not None:
                save_fn(frames, path)

    return {
        "best_video": best_frames,
        "best_score": best_score,
        "best_seed": best_seed,
        "all_scores": all_scores,
        "all_details": all_details,
        "seeds": [s for _, s in candidates],
        "total_time_sec": elapsed,
    }
```

**关键修复点**：
- ✅ 使用 `cosmos3_output_to_pil(result)` 替代硬编码 `output.frames[0]`
- ✅ 生成器设备检测兼容 meta/cpu/cuda（参考 bon_pipeline.py:100-105）
- ✅ 不在 `pipe()` 调用中传 `flow_shift`（应在 scheduler 初始化时设置）

#### 4.2.3 新增 `VGGTomega_Cotracker3/run_bon_vggt_cosmos3.py`
**目的**：提供 Cosmos3 版本的单条测试 CLI

```python
"""
单条 Best-of-N CLI（Cosmos3 + VGGTomega + CoTracker3）。

用法：
    python VGGTomega_Cotracker3/run_bon_vggt_cosmos3.py \
        --model /path/to/Cosmos3-Nano \
        --vggt_model /path/to/vggt_omega_checkpoint.pth \
        --image /path/to/first_frame.png \
        --prompt "robot arm picks up the red block" \
        --N 8 --num_frames 189 --height 720 --width 1280
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from VGGTomega_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGTomega_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig
from VGGTomega_Cotracker3.utils import save_video_from_pil


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoReward VGGT+CoTracker3 Best-of-N Pipeline (Cosmos3 单条)"
    )

    # Cosmos3 生成参数
    parser.add_argument("--model", type=str, required=True,
                        help="Cosmos3 模型路径（Cosmos3-Nano 等）。")
    parser.add_argument("--image", type=str, required=True,
                        help="首帧图片路径。")
    parser.add_argument("--prompt", type=str, required=True,
                        help="动作指令文本。")
    parser.add_argument("--num_frames", type=int, default=189,
                        help="输出帧数。")
    parser.add_argument("--fps", type=int, default=24,
                        help="帧率。")
    parser.add_argument("--height", type=int, default=720,
                        help="输出高度。")
    parser.add_argument("--width", type=int, default=1280,
                        help="输出宽度。")
    parser.add_argument("--num_inference_steps", type=int, default=35,
                        help="去噪步数。")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="CFG scale。")
    parser.add_argument("--flow_shift", type=float, default=10.0,
                        help="Scheduler flow shift。")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="负面提示词。")

    # BoN 参数
    parser.add_argument("--N", type=int, default=8,
                        help="候选数。")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="基础种子（候选使用 seed_base+i）。")

    # VGGTomega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGTomega 权重路径（.pth 文件）。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。None 则用 torch.hub 自动下载。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20,
                        help="评分用帧数。")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="CoTracker3 每批追踪点数。")
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGTomega 输入分辨率（长边参考）。")

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50)
    parser.add_argument("--dynamic_weight", type=float, default=0.30)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/bon_vggt_cosmos3")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 Cosmos3 模型（参考 geo_reward/run_bon_cosmos3.py:191-223）
    logger.info(f"加载 Cosmos3 模型: {args.model}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # ← 使用 bfloat16，与现有代码一致
    )
    pipe = pipe.to("cuda")
    
    # ✅ flow_shift 在 scheduler 上设置，不在 pipe() 调用时传入
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )

    # 2. 加载输入图片
    image = Image.open(args.image).convert("RGB")

    # 3. 配置 Reward
    reward_cfg = VGGTReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        image_size=args.image_resolution,
        chunk_size=args.chunk_size,
        max_frames=args.num_frames_for_reward,
    )

    # 4. 创建 BoN pipeline
    bon = GeoRewardBoNVGGT(
        cosmos3_pipe=pipe,
        vggt_model_path=args.vggt_model,
        cotracker_model_path=args.cotracker_model,
        N=args.N,
        num_frames_for_reward=args.num_frames_for_reward,
        reward_config=reward_cfg,
        chunk_size=args.chunk_size,
    )

    # 5. 执行 BoN 生成（⚠️ 不传 flow_shift）
    logger.info(f"开始 Best-of-N 生成: N={args.N}")
    result = bon.generate(
        image=image,
        prompt=args.prompt,
        N=args.N,
        num_frames=args.num_frames,
        fps=args.fps,
        seed_base=args.seed_base,
        output_dir=args.output_dir,
        save_fn=save_video_from_pil,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        # ⚠️ 不传 flow_shift（已在 scheduler 上设置）
    )

    # 6. 输出结果
    logger.info(f"最优候选: seed={result['best_seed']}, score={result['best_score']:.4f}")
    logger.info(f"所有分数: {result['all_scores']}")
    logger.info(f"总用时: {result['total_time_sec']:.1f}s")

    # 7. 保存结果 JSON
    result_json = {
        "prompt": args.prompt,
        "best_seed": result["best_seed"],
        "best_score": result["best_score"],
        "all_scores": result["all_scores"],
        "all_seeds": result["seeds"],
        "config": vars(args),
    }
    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info(f"结果已保存: {json_path}")


if __name__ == "__main__":
    main()
```

**关键修复点**：
- ✅ 使用 `torch.bfloat16` 替代 `torch.float16`
- ✅ `flow_shift` 在 scheduler 初始化时设置，不在 `pipe()` 调用时传入
- ✅ 导入自定义 `save_video_from_pil`（从 `VGGTomega_Cotracker3.utils`）

#### 4.2.4 新增 `VGGTomega_Cotracker3/run_bon_batch_vggt_cosmos3.py`
**目的**：提供 Cosmos3 版本的批量测试 CLI（参考 `run_batch_bon.py` 的结构）

```python
"""
批量 Best-of-N CLI（Cosmos3 + VGGTomega + CoTracker3）。

用法：
    python VGGTomega_Cotracker3/run_bon_batch_vggt_cosmos3.py \
        --batch_json batch_prompts_inputs_real_6.json \
        --image_dir /path/to/inputs/inputs_real_6 \
        --output_dir outputs/bon_vggt_inputs_real_6 \
        --model /path/to/Cosmos3-Nano \
        --vggt_model /path/to/vggt_omega.pth \
        --N 8
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from VGGTomega_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGTomega_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig
from VGGTomega_Cotracker3.utils import save_video_from_pil


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量 GeoReward VGGT+CoTracker3 Best-of-N Pipeline (Cosmos3)"
    )

    # 批次配置
    parser.add_argument("--batch_json", type=str, required=True,
                        help="JSON 文件路径，映射图片 stem 到 prompt。")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图片目录。")
    parser.add_argument("--image_ext", type=str, default=".jpg",
                        help="图片文件扩展名。")

    # Cosmos3 模型
    parser.add_argument("--model", type=str, required=True,
                        help="Cosmos3 模型路径。")

    # 生成参数
    parser.add_argument("--num_frames", type=int, default=189)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--negative_prompt", type=str, default="")

    # BoN 参数
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--seed_base", type=int, default=42)

    # VGGTomega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGTomega 权重路径。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--image_resolution", type=int, default=512)

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50)
    parser.add_argument("--dynamic_weight", type=float, default=0.30)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/bon_vggt_batch")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 batch JSON
    with open(args.batch_json, "r") as f:
        batch_prompts = json.load(f)

    logger.info(f"加载 {len(batch_prompts)} 个测试用例")

    # 2. 加载 Cosmos3 模型（只加载一次）
    logger.info(f"加载 Cosmos3 模型: {args.model}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # ← 使用 bfloat16
    )
    pipe = pipe.to("cuda")
    
    # ✅ flow_shift 在 scheduler 上设置
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )

    # 3. 配置 Reward
    reward_cfg = VGGTReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        image_size=args.image_resolution,
        chunk_size=args.chunk_size,
        max_frames=args.num_frames_for_reward,
    )

    # 4. 创建 BoN pipeline（模型延迟加载）
    bon = GeoRewardBoNVGGT(
        cosmos3_pipe=pipe,
        vggt_model_path=args.vggt_model,
        cotracker_model_path=args.cotracker_model,
        N=args.N,
        num_frames_for_reward=args.num_frames_for_reward,
        reward_config=reward_cfg,
        chunk_size=args.chunk_size,
    )

    # 5. 批量处理
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    for case_idx, (image_stem, prompt) in enumerate(batch_prompts.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Case {case_idx}/{len(batch_prompts)}: {image_stem}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"{'='*60}")

        # 加载输入图片
        image_path = os.path.join(args.image_dir, image_stem + args.image_ext)
        if not os.path.exists(image_path):
            logger.warning(f"图片不存在: {image_path}，跳过")
            continue

        image = Image.open(image_path).convert("RGB")

        # 执行 BoN 生成
        case_output_dir = os.path.join(args.output_dir, image_stem)
        os.makedirs(case_output_dir, exist_ok=True)

        try:
            result = bon.generate(
                image=image,
                prompt=prompt,
                N=args.N,
                num_frames=args.num_frames,
                fps=args.fps,
                seed_base=args.seed_base,
                output_dir=case_output_dir,
                save_fn=save_video_from_pil,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                # ⚠️ 不传 flow_shift
            )

            logger.info(f"最优: seed={result['best_seed']}, score={result['best_score']:.4f}")

            # 保存结果
            case_result = {
                "case": image_stem,
                "prompt": prompt,
                "best_seed": result["best_seed"],
                "best_score": result["best_score"],
                "all_scores": result["all_scores"],
                "all_seeds": result["seeds"],
                "time_sec": result["total_time_sec"],
            }
            all_results.append(case_result)

            # 保存 case 级别 JSON
            with open(os.path.join(case_output_dir, "result.json"), "w") as f:
                json.dump(case_result, f, indent=2)

        except Exception as e:
            logger.error(f"Case {image_stem} 失败: {e}", exc_info=True)
            continue

    # 6. 保存汇总结果
    summary = {
        "total_cases": len(batch_prompts),
        "completed_cases": len(all_results),
        "results": all_results,
        "config": vars(args),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n批量处理完成: {len(all_results)}/{len(batch_prompts)} 成功")
    logger.info(f"汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
```

#### 4.2.5 新增 `VGGTomega_Cotracker3/utils.py`
**目的**：VGGTomega_Cotracker3 专用的工具函数（与 geo_reward 解耦）

```python
"""
VGGTomega_Cotracker3 专用工具函数。
"""

import os
import cv2
import numpy as np
from PIL import Image


def save_video_from_pil(frames_pil, path, fps=24):
    """
    从 PIL 图片列表保存视频。
    
    Args:
        frames_pil: List[PIL.Image]
        path: 输出路径（.mp4）
        fps: 帧率
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # 获取视频尺寸
    first_frame = np.array(frames_pil[0])
    height, width = first_frame.shape[:2]
    
    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames_pil:
        frame_np = np.array(frame)
        # PIL 是 RGB，cv2 需要 BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
```

**注意**：
- ❌ **不要修改 `geo_reward/utils.py`**，保持与现有 4RC 流程完全解耦
- ✅ `cosmos3_output_to_pil` 已存在于 `geo_reward/utils.py`，直接复用即可

---

## 5. 使用示例

### 5.1 单条测试
```bash
cd /path/to/now_cosmos3

python VGGTomega_Cotracker3/run_bon_vggt_cosmos3.py \
    --model /path/to/Cosmos3-Nano \
    --vggt_model /path/to/vggt_omega_checkpoint.pth \
    --image /path/to/test_image.jpg \
    --prompt "A robot arm picks up the red block" \
    --N 8 \
    --num_frames 189 \
    --height 720 \
    --width 1280 \
    --num_inference_steps 35 \
    --guidance_scale 6.0 \
    --output_dir outputs/bon_vggt_test
```

### 5.2 批量测试
```bash
cd /path/to/now_cosmos3

python VGGTomega_Cotracker3/run_bon_batch_vggt_cosmos3.py \
    --batch_json batch_prompts_inputs_real_6.json \
    --image_dir /path/to/inputs/inputs_real_6 \
    --output_dir outputs/bon_vggt_inputs_real_6 \
    --model /path/to/Cosmos3-Nano \
    --vggt_model /path/to/vggt_omega.pth \
    --N 8 \
    --num_frames 189 \
    --height 720 \
    --width 1280
```

### 5.3 Python API 使用
```python
from PIL import Image
import torch
from diffusers import Cosmos3OmniPipeline

from VGGTomega_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGTomega_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig
from geo_reward.utils import save_video_from_pil

# 1. 加载 Cosmos3 模型
pipe = Cosmos3OmniPipeline.from_pretrained(
    "/path/to/Cosmos3-Nano",
    torch_dtype=torch.float16,
).to("cuda")

# 2. 加载输入图片
image = Image.open("test.jpg").convert("RGB")

# 3. 配置 Reward
reward_cfg = VGGTReconRewardConfig(
    static_weight=0.50,
    dynamic_weight=0.30,
    motion_weight=0.20,
    image_size=512,
    chunk_size=4096,
    max_frames=20,
)

# 4. 创建 BoN pipeline
bon = GeoRewardBoNVGGT(
    cosmos3_pipe=pipe,
    vggt_model_path="/path/to/vggt_omega.pth",
    cotracker_model_path=None,  # 使用 torch.hub 自动下载
    N=8,
    num_frames_for_reward=20,
    reward_config=reward_cfg,
)

# 5. 执行 BoN 生成
result = bon.generate(
    image=image,
    prompt="A robot arm picks up the red block",
    N=8,
    num_frames=189,
    fps=24,
    seed_base=42,
    output_dir="outputs/test",
    save_fn=save_video_from_pil,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
)

print(f"最优 seed: {result['best_seed']}")
print(f"最优分数: {result['best_score']:.4f}")
print(f"所有分数: {result['all_scores']}")
```

---

## 6. 验证与测试

### 6.1 功能验证清单
- [ ] VGGTomega 模型加载正常
- [ ] CoTracker3 模型加载正常（torch.hub 和本地权重两种方式）
- [ ] 组合推理输出格式正确（pts, track, extrinsic, intrinsic, visibility）
- [ ] GeoReward 计算无错误，分数在 [0, 1] 范围内
- [ ] Best-of-N pipeline 正常运行，选出最优候选
- [ ] 显存管理正常，无 OOM
- [ ] 单条测试 CLI 运行正常
- [ ] 批量测试 CLI 运行正常

### 6.2 对比测试
在相同的测试集上运行 4RC 和 VGGT 两种后端，对比：
1. **评分分布**：两种后端的 reward 分数分布是否一致
2. **排名一致性**：对于同一组候选，两种后端选出的最优候选是否相同
3. **计算效率**：推理时间、显存占用
4. **评分稳定性**：同一视频多次评分的一致性

### 6.3 测试脚本示例
```python
# test_vggt_vs_4rc.py
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline

# 4RC 后端
from geo_reward.bon_pipeline import Cosmos3GeoRewardBoN
from geo_reward.fourrc_adapter import load_fourrc_model
from geo_reward.recon_reward import ReconstructionReward, ReconRewardConfig

# VGGT 后端
from VGGTomega_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGTomega_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig

# 加载模型
pipe = Cosmos3OmniPipeline.from_pretrained(...).to("cuda")
fourrc_model = load_fourrc_model(...)
vggt_path = "/path/to/vggt_omega.pth"

# 测试图片
image = Image.open("test.jpg")
prompt = "A robot arm picks up the red block"

# 生成候选视频（使用固定 seed）
seeds = [42, 43, 44, 45, 46, 47, 48, 49]
candidates = []
for seed in seeds:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(prompt=prompt, image=image, generator=gen, ...)
    candidates.append(output.frames[0])

# 用 4RC 后端评分
reward_4rc = ReconstructionReward(fourrc_model=fourrc_model, ...)
scores_4rc = []
for frames in candidates:
    r = reward_4rc.compute_reward(frames)
    scores_4rc.append(r["total"])

# 用 VGGT 后端评分
reward_vggt = VGGTReconstructionReward(vggt_model_path=vggt_path, ...)
scores_vggt = []
for frames in candidates:
    r = reward_vggt.compute_reward(frames)
    scores_vggt.append(r["total"])

# 对比
print("4RC 分数:", scores_4rc)
print("VGGT 分数:", scores_vggt)
print("4RC 最优:", seeds[scores_4rc.index(max(scores_4rc))])
print("VGGT 最优:", seeds[scores_vggt.index(max(scores_vggt))])

# 排名相关性
from scipy.stats import spearmanr
corr, p_value = spearmanr(scores_4rc, scores_vggt)
print(f"Spearman 相关系数: {corr:.4f}, p-value: {p_value:.4f}")
```

---

## 7. 依赖与环境

### 7.1 新增依赖
```txt
# VGGTomega 相关
torchvision

# CoTracker3 相关
# 方式 1: torch.hub 自动下载（无需手动安装）
# 方式 2: 手动安装 co-tracker 代码

# 视频保存（save_video_from_pil）
opencv-python  # ← P1: 必须添加，否则 cv2 导入失败
```

### 7.2 环境配置
```bash
# 假设已有 now_cosmos3 的基础环境
cd /path/to/now_cosmos3

# 确认 VGGTomega 代码可导入
export PYTHONPATH="${PYTHONPATH}:/path/to/now_cosmos3/VGGTomega/vggt-omega-main"

# 确认 CoTracker3 代码可导入
export PYTHONPATH="${PYTHONPATH}:/path/to/now_cosmos3/co-tracker-main/co-tracker-main"

# 或者在代码中动态添加（适配器中已实现）
```

### 7.3 模型权重
- **VGGTomega**: 需要下载预训练权重 `.pth` 文件
- **CoTracker3**: 可选
  - 方式 1: 使用 `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")` 自动下载
  - 方式 2: 手动下载权重文件，通过 `checkpoint_path` 参数传入

---

## 8. 注意事项

### 8.1 显存管理
- **生成阶段**：Cosmos3 DiT + VAE 在 GPU
- **评分阶段**：VGGTomega + CoTracker3 在 GPU，DiT + VAE 卸载到 CPU
- **分 chunk 处理**：CoTracker3 的 `chunk_size` 参数控制每批追踪点数
  - 720p 视频：建议 `chunk_size=4096`
  - 1080p 视频：建议 `chunk_size=2048`
  - 如遇 OOM，减小 `chunk_size`

### 8.2 坐标系一致性
- VGGTomega 输出 **camera-from-world** (3, 4)，需转换为 **camera-to-world** (4, 4)
- CoTracker3 tracks 格式 **(x, y)**，x=列坐标，y=行坐标
- 反投影时使用 VGGTomega 的 intrinsic 和 extrinsic

### 8.3 分辨率对齐
- CoTracker3 输入帧需 resize 到与 VGGTomega 相同的分辨率
- VGGTomega 使用 balanced 模式，不同帧可能有不同分辨率，需统一 padding

### 8.4 数值稳定性
- Depth 值 clamp 到 `min=0.01`，避免除零
- 替换 NaN/Inf 为 0
- 不可见像素的 track 设为 0

---

## 9. 预期成果

### 9.1 技术成果
1. **通用性验证**：证明 GeoReward 方法可适配不同重建模型
2. **模块化架构**：清晰的适配器模式，便于未来集成更多重建模型
3. **对比基准**：4RC vs VGGTomega+CoTracker3 的评分对比数据

### 9.2 实验数据
在 inputs_real_6 测试集上运行两种后端，产出：
- 每个 case 的 reward 分数对比
- 排名一致性分析（Spearman 相关系数）
- 计算效率对比（推理时间、显存占用）
- 最优候选的视觉质量对比

### 9.3 文档产出
- 集成方案文档（本文档）
- 使用教程和 API 文档
- 对比实验报告

---

## 10. 后续扩展方向

### 10.1 更多重建模型
- **DepthAnything**: 单目深度估计 SOTA
- **MoGe**: 单目几何重建（depth + normal）
- **DUSt3R**: 双目立体重建
- **PixelSplat**: 3D Gaussian Splatting 重建

### 10.2 高级 BoN 策略
- **渐进淘汰 BoN**: 在去噪中间 checkpoint 评分，提前淘汰低分候选
- **树分支 BoN**: 少量 trunk 共享前段去噪，分支点注入噪声
- **梯度引导**: 去噪过程中反向传播 reward loss 引导 latent
  - **注意**: VGGTomega 是预训练模型，梯度引导需要额外设计

### 10.3 多模态融合
- 组合多个重建模型的输出（4RC + VGGTomega + DepthAnything）
- 加权融合 reward 分数
- 集成学习方法

---

## 11. 总结

本方案提供了将 **VGGTomega + CoTracker3** 重建方法集成到 **now_cosmos3** 项目的完整路径：

1. **已完成**：复制 VGGTomega、co-tracker-main、VGGTomega_Cotracker3 三个文件夹
2. **核心修改**：适配 Cosmos3 的 I2V 模型接口
   - 修改 `bon_pipeline_vggt.py` 中的模型调用
   - 新增 Cosmos3 专用的 CLI 脚本
   - 添加 Cosmos3 工具函数到 `utils.py`
3. **验证测试**：单条测试 → 批量测试 → 对比实验
4. **产出文档**：集成方案 + 使用教程 + 对比报告

**关键优势**：
- **模块化设计**：适配器模式，便于扩展
- **代码复用**：VGGTomega_Cotracker3 核心逻辑不变，只修改 I2V 接口
- **显存优化**：DiT/VAE 与 重建模型 交替占用 GPU
- **通用性强**：为未来集成更多重建模型提供架构参考

审查通过后，即可开始代码修改实施。
