import numpy as np
from PIL import Image
import torch


def sample_frames(total_frames=81, max_frames=20):
    """
    Uniformly sample frame indices, always including the first frame.

    Args:
        total_frames: Total number of frames in the video.
        max_frames: Maximum number of frames to sample.

    Returns:
        Sorted list of unique frame indices.
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
    - Pipeline result object with .video / .frames / .images attribute
    - List of PIL Images (default output_type="pil")
    - List of numpy arrays
    - Tensor (output_type="pt"): (B, T, C, H, W) / (T, C, H, W) / (C, T, H, W)
    - Nested list (batch wrapper): [[PIL, PIL, ...]]
    - Numpy array (T, H, W, C)
    """
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
            return [
                Image.fromarray(f) if f.dtype == np.uint8
                else Image.fromarray((f * 255).astype(np.uint8))
                for f in frames
            ]

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
            return [
                Image.fromarray(f) if frames.dtype == np.uint8
                else Image.fromarray((f * 255).astype(np.uint8))
                for f in frames
            ]

    raise ValueError(
        f"Unsupported Cosmos3 output format: type={type(frames)}, "
        f"shape={getattr(frames, 'shape', None)}"
    )


def transform_to_camera(pts_world, w2c):
    """
    Transform world-coordinate points to camera coordinates.

    Args:
        pts_world: (..., 3) world-coordinate points (torch tensor).
        w2c: (4, 4) world-to-camera transform.

    Returns:
        (..., 3) points in camera coordinates.
    """
    original_shape = pts_world.shape
    pts_flat = pts_world.reshape(-1, 3)
    pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
    pts_cam = (w2c @ pts_homo.T).T[:, :3]
    return pts_cam.reshape(original_shape)
