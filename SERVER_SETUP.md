# now_cosmos3 服务器部署文档

## 1. 服务器环境信息

| 项目 | 信息 |
|------|------|
| 服务器地址 | aibox-r915b2f930e6-696888777b-9l2fc |
| GPU | 4x NVIDIA RPBZZZ6 (每卡 ~98GB VRAM) |
| CUDA 版本 | 13.0 (Driver 580.126.20) |
| PyTorch 版本 | 2.11.0+cu128 |
| Conda 环境 | `now_cosmos3` (Python 3.11) |
| 代理 | `http://10.184.64.107:18000` |

## 2. 目录结构

```
/pfs/mayuema/spj/now_cosmos3/              # 项目根目录
├── cosmos-main/
│   └── cosmos-main/                       # Cosmos3 官方仓库 (文档/评估/cookbooks)
│       ├── README.md                      # 官方使用说明 (含各类推理方式)
│       ├── cookbooks/cosmos3/             # 各类使用示例
│       │   ├── generator/                 # Generator 相关 (视频/图片/声音/动作生成)
│       │   │   ├── action/               # 动作生成/策略推理
│       │   │   ├── audiovisual/          # 视听生成
│       │   │   └── transfer/             # Transfer 控制生成
│       │   └── reasoner/                  # Reasoner 相关 (视觉理解/推理)
│       │       └── assets/               # 示例图片/视频素材
│       └── evaluation/                    # 评估代码
│
├── 4RC-main/
│   └── 4RC-main/                          # 4RC 4D重建模型代码
│       ├── app.py                         # Gradio Web Demo
│       ├── inference.py                   # CLI 推理脚本
│       ├── arc/                           # 核心模型代码
│       │   ├── models/arc/               # ARC 模型定义
│       │   ├── dust3r/                   # DUSt3R 相关 (3D重建)
│       │   └── croco/                    # CroCo 预训练模型
│       └── requirements.txt
│
├── geo_reward/                            # GeoReward 奖励模块 (自研)
│   ├── recon_reward.py                   # 重建质量奖励计算
│   ├── fourrc_adapter.py                 # 4RC 模型接口适配器
│   ├── bon_pipeline.py                   # BoN 选择管线
│   ├── run_bon_cosmos3.py                # Cosmos3 + BoN 运行脚本
│   └── utils.py                          # 帧转换工具
│
├── Cosmos3-Nano/                          # Cosmos3-Nano 16B 模型权重 (~35GB)
│
├── GeoReward_Integration_Plan.md          # GeoReward 集成方案文档
├── SERVER_SETUP.md                        # 本文档
└── .gitignore
```

## 3. 模型权重路径

| 模型 | 路径 | 大小 |
|------|------|------|
| Cosmos3-Nano (16B) | `/pfs/mayuema/spj/now_cosmos3/Cosmos3-Nano` | ~35GB |
| 4RC | `/pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC` | - |

## 4. 环境激活与代理

每次登录服务器后执行：

```bash
conda activate now_cosmos3
export http_proxy=http://10.184.64.107:18000
export https_proxy=http://10.184.64.107:18000
```

如果已写入 `~/.bashrc` 则代理自动生效，只需激活 conda 环境。

## 5. 推理代码示例

### 5.1 Cosmos3 Generator — Text-to-Video (Diffusers)

```python
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "/pfs/mayuema/spj/now_cosmos3/Cosmos3-Nano",  # 本地权重路径
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)

result = pipe(
    prompt="A mobile robot navigates a warehouse aisle and stops at a shelf.",
    negative_prompt="",
    image=None,
    num_frames=189,      # ~7.9秒 @24FPS
    height=720,
    width=1280,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)

export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

### 5.2 Cosmos3 Generator — Image-to-Video (Diffusers)

```python
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "/pfs/mayuema/spj/now_cosmos3/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)

image = Image.open("input_image.jpg")

result = pipe(
    prompt="The robot arm picks up the object smoothly.",
    negative_prompt="blurry, distorted",
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

### 5.3 Cosmos3 Reasoner — 图片/视频理解 (Transformers)

```python
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "/pfs/mayuema/spj/now_cosmos3/Cosmos3-Nano"

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "your_image.jpg"},
            {"type": "text", "text": "Caption the image in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 5.4 4RC — CLI 推理 (3D 重建)

```bash
cd /pfs/mayuema/spj/now_cosmos3/4RC-main/4RC-main

# 从图片文件夹推理
python inference.py \
    --input /path/to/image_folder \
    --save output.npz \
    --checkpoint_dir /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC

# 从视频文件推理
python inference.py \
    --input /path/to/video.mp4 \
    --save output.npz \
    --checkpoint_dir /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC
```

### 5.5 4RC — Gradio Web Demo

```bash
cd /pfs/mayuema/spj/now_cosmos3/4RC-main/4RC-main

python app.py \
    --checkpoint_dir /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
    --output_dir ./demo_outputs
```

## 6. 同步工作流

```
本地电脑 (Windows)          GitHub                    远程服务器 (Linux)
     git push         -->   Jie-ser/now_cosmos3  <--    git pull
```

```bash
# 本地推送
git add .
git commit -m "xxx"
git push

# 服务器拉取
cd /pfs/mayuema/spj/now_cosmos3
export https_proxy=http://10.184.64.107:18000
git pull
```

## 7. 关键参数参考

### Cosmos3 Generator 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_frames` | 帧数 (5~300) | 189 |
| `height` x `width` | 分辨率 (256p/480p/720p) | 720x1280 |
| `fps` | 帧率 (10/16/24/30) | 24 |
| `num_inference_steps` | 去噪步数 | 35 |
| `guidance_scale` | CFG 引导强度 | 6.0 |
| `flow_shift` | Scheduler flow-shift | 10.0 |
| `enable_sound` | 是否生成声音 | False |

### 4RC 推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入图片文件夹或视频 | 必填 |
| `--save` | 输出 .npz 路径 | 必填 |
| `--checkpoint_dir` | 模型权重路径 | `Luo-Yihang/4RC` |
| `--track_query_idx` | 追踪查询帧索引 (-1=中间帧) | -1 |
| `--refine_track_visualization` | 精细化追踪可视化 | False |

## 8. 注意事项

1. **GPU 选择**：GPU 0 可能被其他任务占用，推理时可通过 `CUDA_VISIBLE_DEVICES=1` 指定空闲 GPU
2. **显存**：Cosmos3-Nano 16B 在 BF16 下约需 32-40GB 显存，单卡可运行
3. **首次运行**：Diffusers 首次加载模型会较慢（需要转换权重格式），后续会使用缓存
4. **模型权重不要 git 同步**：`Cosmos3-Nano/` 目录约 35GB，已在 `.gitignore` 中排除（如果没有请手动添加）
