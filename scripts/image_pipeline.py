#!/usr/bin/env python3
"""Shared image preprocessing and Sun raster export helpers."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUN_RASTER_MAGIC = 0x59A66A95
RASTER_TYPE_STANDARD = 1
MAP_TYPE_NONE = 0
MAP_LENGTH = 0


def resize_short_side_bgr(image_bgr: np.ndarray, short_side: int) -> np.ndarray | None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    height, width = image_bgr.shape[:2]
    if height < 2 or width < 2:
        return None

    scale = float(short_side) / min(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    if resized_width < short_side or resized_height < short_side:
        return None

    return cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"cannot crop {size}x{size} from {width}x{height}")
    left = (width - size) // 2
    top = (height - size) // 2
    return image[top : top + size, left : left + size]


def crop_256_bgr(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    resized = resize_short_side_bgr(image, 256)
    if resized is None:
        return None
    return center_crop(resized, 256)


def read_ras_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{path}: expected three-channel image, got shape {image.shape}")
    return image


def preprocess_bgr_for_resnet(image_bgr: np.ndarray) -> np.ndarray:
    resized = resize_short_side_bgr(image_bgr, 256)
    if resized is None:
        raise ValueError("expected readable HWC BGR image with three channels")
    crop = center_crop(resized, 224)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1).astype("<f4", copy=False)


def preprocess_path(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return preprocess_bgr_for_resnet(image)


def _sun_raster_header(width: int, height: int) -> bytes:
    return struct.pack(
        ">8I",
        SUN_RASTER_MAGIC,
        width,
        height,
        24,
        width * height * 3,
        RASTER_TYPE_STANDARD,
        MAP_TYPE_NONE,
        MAP_LENGTH,
    )


def write_ras_bgr(path: Path, image_bgr: np.ndarray) -> None:
    height, width, channels = image_bgr.shape
    if channels != 3:
        raise ValueError("expected HWC BGR image")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_sun_raster_header(width, height) + image_bgr.tobytes())


def write_ras_rgb(path: Path, image_rgb: np.ndarray) -> None:
    height, width, channels = image_rgb.shape
    if channels != 3:
        raise ValueError("expected HWC RGB image")
    image_bgr = image_rgb[..., [2, 1, 0]]
    write_ras_bgr(path, np.ascontiguousarray(image_bgr))


def make_rgb_color_card(block_size: int = 64) -> np.ndarray:
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]
    card = np.empty((block_size, block_size * len(colors), 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        left = index * block_size
        card[:, left : left + block_size] = color
    return card
