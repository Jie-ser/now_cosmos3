"""
VGGTomega_Cotracker3 专用工具函数。
"""

import os
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
    # 延迟导入 cv2（避免缺依赖时整个包导入失败）
    import cv2

    # 确保输出目录存在
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

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
