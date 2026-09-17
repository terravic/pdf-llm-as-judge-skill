"""utils/comb_filter.py

ROI cropping and comb-line artifact suppression for segmented form fields.
Provides dual-engine execution using OpenCV if available, with native
Pillow + NumPy fallback for environments without OpenCV installed.
"""

from __future__ import annotations

import io
from typing import Tuple, Union
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image, ImageOps


def load_image(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Loads an image from a file path, byte buffer, or NumPy array into BGR format."""
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, bytes):
        if HAS_CV2:
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        pil_img = Image.open(io.BytesIO(source)).convert("RGB")
        rgb = np.array(pil_img)
        return rgb[:, :, ::-1].copy()
    if isinstance(source, str):
        if HAS_CV2:
            img = cv2.imread(source)
            if img is not None:
                return img
        pil_img = Image.open(source).convert("RGB")
        rgb = np.array(pil_img)
        return rgb[:, :, ::-1].copy()
    if isinstance(source, Image.Image):
        pil_img = source.convert("RGB")
        rgb = np.array(pil_img)
        return rgb[:, :, ::-1].copy()
    raise TypeError(f"Unsupported image source type: {type(source)}")


def crop_field_roi(
    image: Union[np.ndarray, Image.Image, str, bytes],
    bbox_norm: Tuple[float, float, float, float],
    padding_px: int = 15,
    padding_ratio: Optional[float] = None,
) -> np.ndarray:
    """Extracts a high-resolution sub-image from a full-page image using normalized coordinates.

    Args:
        image: Full-page image as BGR NumPy array, PIL Image, file path, or bytes.
        bbox_norm: Tuple of (ymin, xmin, ymax, xmax) normalized to [0.0, 1.0].
        padding_px: Pixel buffer added around boundary to preserve outer strokes.
        padding_ratio: Optional fractional buffer (e.g. 0.03 for 3%). Overrides padding_px if set.

    Returns:
        High-resolution cropped BGR NumPy array.
    """
    img_bgr = load_image(image) if not isinstance(image, np.ndarray) else image
    h, w = img_bgr.shape[:2]
    ymin, xmin, ymax, xmax = bbox_norm

    if padding_ratio is not None:
        pad_y = int(h * padding_ratio)
        pad_x = int(w * padding_ratio)
    else:
        pad_y = pad_x = padding_px

    y1 = max(0, int(ymin * h) - pad_y)
    x1 = max(0, int(xmin * w) - pad_x)
    y2 = min(h, int(ymax * h) + pad_y)
    x2 = min(w, int(xmax * w) + pad_x)

    return img_bgr[y1:y2, x1:x2].copy()


def suppress_comb_lines(
    crop_bgr: Union[np.ndarray, Image.Image],
    saturation_threshold: int = 28,
    value_threshold: int = 225,
) -> np.ndarray:
    """Suppresses pre-printed black/gray form tick marks while isolating colored pen ink.

    Falls back safely to contrast-enhanced grayscale if ink is black.

    Args:
        crop_bgr: Cropped BGR field image or PIL Image.
        saturation_threshold: Saturation boundary below which neutral template marks are filtered.
        value_threshold: Value/brightness ceiling for ink detection.

    Returns:
        Cleaned BGR image ready for multimodal inference.
    """
    img_bgr = load_image(crop_bgr) if not isinstance(crop_bgr, np.ndarray) else crop_bgr

    if HAS_CV2:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        ink_mask = (s > saturation_threshold) & (v < value_threshold)

        # Fallback for black ballpoint ink: apply CLAHE contrast enhancement
        if np.sum(ink_mask) < 30:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        # Project ink onto a clean white background
        cleaned = np.full_like(img_bgr, 255)
        cleaned[ink_mask] = img_bgr[ink_mask]
        return cleaned

    # Native Pillow + NumPy implementation
    rgb = img_bgr[:, :, ::-1]
    pil_rgb = Image.fromarray(rgb)
    hsv_arr = np.array(pil_rgb.convert("HSV"))
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]
    ink_mask = (s > saturation_threshold) & (v < value_threshold)

    if np.sum(ink_mask) < 30:
        gray = pil_rgb.convert("L")
        enhanced = ImageOps.autocontrast(gray)
        enhanced_bgr = np.array(enhanced.convert("RGB"))[:, :, ::-1]
        return enhanced_bgr

    cleaned = np.full_like(img_bgr, 255)
    cleaned[ink_mask] = img_bgr[ink_mask]
    return cleaned


def prepare_crop_payload(
    full_image_bgr: Union[np.ndarray, str, bytes],
    bbox_norm: Tuple[float, float, float, float],
    padding_px: int = 15,
) -> bytes:
    """Crops, filters, and encodes the field to PNG bytes for LLM payload."""
    img_bgr = load_image(full_image_bgr)
    raw_crop = crop_field_roi(img_bgr, bbox_norm, padding_px=padding_px)
    filtered_crop = suppress_comb_lines(raw_crop)

    if HAS_CV2:
        success, buffer = cv2.imencode(
            ".png", filtered_crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        if not success:
            raise ValueError("Failed to encode cropped image to PNG buffer.")
        return buffer.tobytes()

    # Pillow encoding fallback
    rgb = filtered_crop[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()
