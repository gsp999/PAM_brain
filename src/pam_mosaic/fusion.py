"""Warp original floating-point frames into a shared mosaic canvas."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from .poses import pose_to_matrix


def _canvas(poses: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    h, w = shape
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]]).T
    all_corners = np.concatenate([(pose_to_matrix(p) @ corners)[:2].T for p in poses])
    low = np.floor(all_corners.min(axis=0)).astype(int)
    high = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = (high - low + 1).tolist()
    if width * height > 150_000_000:
        raise ValueError(f"Mosaic canvas is too large: {width} x {height}")
    offset = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], dtype=float)
    return width, height, offset


def _canvas_varied(poses: np.ndarray, shapes: list[tuple[int, int]]) -> tuple[int, int, np.ndarray]:
    all_corners = []
    for pose, (h, w) in zip(poses, shapes):
        corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]]).T
        all_corners.append((pose_to_matrix(pose) @ corners)[:2].T)
    coordinates = np.concatenate(all_corners)
    low = np.floor(coordinates.min(axis=0)).astype(int)
    high = np.ceil(coordinates.max(axis=0)).astype(int)
    width, height = (high - low + 1).tolist()
    if width * height > 150_000_000:
        raise ValueError(f"Mosaic canvas is too large: {width} x {height}")
    offset = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], dtype=float)
    return width, height, offset


def fuse(frames: list[np.ndarray], poses: np.ndarray, output: Path,
         *, display_lo: float = 0.02, display_hi: float = 0.25) -> tuple[int, int]:
    if not frames:
        raise ValueError("No frames to fuse")
    width, height, offset = _canvas(poses, frames[0].shape)
    weighted_sum = np.zeros((height, width), np.float64)
    weight_sum = np.zeros((height, width), np.float32)
    h, w = frames[0].shape
    valid = np.pad(np.ones((h, w), np.uint8), 1)
    feather = np.minimum(distance_transform_edt(valid)[1:-1, 1:-1], 35).astype(np.float32)
    feather = np.maximum(feather, 1) / 35
    for frame, pose in zip(frames, poses):
        matrix = (offset @ pose_to_matrix(pose))[:2].astype(np.float32)
        image = cv2.warpAffine(frame.astype(np.float32), matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderValue=0)
        weight = cv2.warpAffine(feather, matrix, (width, height),
                                flags=cv2.INTER_LINEAR, borderValue=0)
        weighted_sum += image * weight
        weight_sum += weight
    mosaic = np.divide(weighted_sum, weight_sum, out=np.zeros_like(weighted_sum), where=weight_sum > 0)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mosaic_float64.npy", mosaic)
    np.save(output / "weight_map.npy", weight_sum)
    preview = np.uint8(np.rint(255 * np.clip((mosaic - display_lo) / (display_hi - display_lo), 0, 1)))
    cv2.imwrite(str(output / "mosaic_preview.png"), cv2.applyColorMap(preview, cv2.COLORMAP_HOT))
    return width, height


def fuse_tiles(images: list[np.ndarray], weights: list[np.ndarray], poses: np.ndarray,
               output: Path, display_lo: float = .02, display_hi: float = .25) -> tuple[int, int]:
    width, height, offset = _canvas_varied(poses, [image.shape for image in images])
    weighted_sum = np.zeros((height, width), np.float64)
    weight_sum = np.zeros((height, width), np.float32)
    for image, weight, pose in zip(images, weights, poses):
        matrix = (offset @ pose_to_matrix(pose))[:2].astype(np.float32)
        projected_image = cv2.warpAffine(image.astype(np.float32), matrix, (width, height),
                                         flags=cv2.INTER_LINEAR, borderValue=0)
        projected_weight = cv2.warpAffine(weight.astype(np.float32), matrix, (width, height),
                                          flags=cv2.INTER_LINEAR, borderValue=0)
        weighted_sum += projected_image * projected_weight
        weight_sum += projected_weight
    result = np.divide(weighted_sum, weight_sum, out=np.zeros_like(weighted_sum), where=weight_sum > 0)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mosaic_float64.npy", result)
    np.save(output / "weight_map.npy", weight_sum)
    preview = np.uint8(np.rint(255 * np.clip((result-display_lo)/(display_hi-display_lo), 0, 1)))
    cv2.imwrite(str(output / "mosaic_preview.png"), cv2.applyColorMap(preview, cv2.COLORMAP_HOT))
    return width, height
