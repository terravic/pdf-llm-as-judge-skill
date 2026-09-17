"""utils/comb_filter.py

ROI cropping and comb-line artifact suppression for segmented form fields.
Provides dual-engine execution using OpenCV if available, with native
Pillow + NumPy fallback for environments without OpenCV installed.
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image, ImageOps


def load_image(source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Loads an image from a file path, byte buffer, PIL Image, or NumPy array into BGR format."""
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


def suppress_vertical_ticks(
    img_bgr: np.ndarray,
    min_tick_height: int = 7,
    max_tick_width: int = 2,
) -> np.ndarray:
    """Suppresses thin vertical tick marks and dividers using morphological structure filtering.

    Operates on both black ink and gray/black template lines. Differentiates thin vertical
    linear segments (comb ticks and cell dividers) from curved and thick pen strokes.

    Args:
        img_bgr: Field image as BGR NumPy array.
        min_tick_height: Minimum pixel height for vertical line segment to qualify as tick.
        max_tick_width: Maximum pixel width for thin vertical tick marks.

    Returns:
        Cleaned BGR image with vertical tick marks bleached to white (255).
    """
    if HAS_CV2:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_tick_height))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max_tick_width + 1, 1))
        v_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        h_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        pure_ticks = cv2.bitwise_and(v_open, cv2.bitwise_not(h_open))
        cleaned = img_bgr.copy()
        cleaned[pure_ticks > 0] = [255, 255, 255]
        return cleaned

    # Native NumPy vector implementation
    gray = np.mean(img_bgr, axis=2).astype(np.uint8)
    is_dark = gray < 210
    h, w = is_dark.shape

    # Vertical run lengths
    v_run = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        v_run[r] = np.where(is_dark[r], v_run[r - 1] + 1 if r > 0 else 1, 0)
    for r in range(h - 2, -1, -1):
        v_run[r] = np.where(is_dark[r] & (v_run[r + 1] > v_run[r]), v_run[r + 1], v_run[r])

    # Horizontal run lengths
    h_run = np.zeros((h, w), dtype=np.int32)
    for c in range(w):
        h_run[:, c] = np.where(is_dark[:, c], h_run[:, c - 1] + 1 if c > 0 else 1, 0)
    for c in range(w - 2, -1, -1):
        h_run[:, c] = np.where(is_dark[:, c] & (h_run[:, c + 1] > h_run[:, c]), h_run[:, c + 1], h_run[:, c])

    tick_mask = (v_run >= min_tick_height) & (h_run <= max_tick_width)
    cleaned = img_bgr.copy()
    cleaned[tick_mask] = 255
    return cleaned


def suppress_comb_lines(
    crop_bgr: Union[np.ndarray, Image.Image],
    saturation_threshold: int = 28,
    value_threshold: int = 225,
    min_tick_height: int = 7,
    max_tick_width: int = 2,
) -> np.ndarray:
    """Suppresses pre-printed black/gray form tick marks while isolating pen ink.

    Handles both colored pen ink (e.g., blue/purple) and black ballpoint ink.

    Args:
        crop_bgr: Cropped BGR field image or PIL Image.
        saturation_threshold: Saturation boundary for colored ink detection.
        value_threshold: Value/brightness ceiling for ink detection.
        min_tick_height: Minimum pixel height to classify as vertical tick.
        max_tick_width: Maximum pixel width for thin vertical tick marks.

    Returns:
        Cleaned BGR image ready for multimodal inference.
    """
    img_bgr = load_image(crop_bgr) if not isinstance(crop_bgr, np.ndarray) else crop_bgr

    if HAS_CV2:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
    else:
        rgb = img_bgr[:, :, ::-1]
        pil_rgb = Image.fromarray(rgb)
        hsv_arr = np.array(pil_rgb.convert("HSV"))
        s = hsv_arr[:, :, 1]
        v = hsv_arr[:, :, 2]

    colored_ink_mask = (s > saturation_threshold) & (v < value_threshold)

    # 1. Colored ink path (e.g. blue ballpoint / gel pen): project onto white
    if np.sum(colored_ink_mask) >= 30:
        cleaned = np.full_like(img_bgr, 255)
        cleaned[colored_ink_mask] = img_bgr[colored_ink_mask]
        return suppress_vertical_ticks(cleaned, min_tick_height=min_tick_height, max_tick_width=max_tick_width)

    # 2. Black / monochrome ink path: apply morphological directional filtering
    cleaned = suppress_vertical_ticks(img_bgr, min_tick_height=min_tick_height, max_tick_width=max_tick_width)
    return cleaned


def auto_detect_comb_rois(
    image: Union[np.ndarray, Image.Image, str, bytes],
    min_width_ratio: float = 0.08,
) -> List[Tuple[float, float, float, float]]:
    """Automatically locates candidate segmented comb box or grid ROIs on a form image.

    Detects horizontal baselines with periodic vertical tick marks or cell dividers.

    Args:
        image: Source form image as BGR array, PIL Image, file path, or bytes.
        min_width_ratio: Minimum fractional width across the page for a candidate comb row.

    Returns:
        List of normalized bounding boxes (ymin, xmin, ymax, xmax).
    """
    img_bgr = load_image(image) if not isinstance(image, np.ndarray) else image
    h, w = img_bgr.shape[:2]
    gray = np.mean(img_bgr, axis=2).astype(np.uint8)
    is_dark = gray < 180

    rois: List[Tuple[float, float, float, float]] = []
    row_density = np.mean(is_dark, axis=1)

    in_row = False
    start_r = 0
    for r in range(h):
        if row_density[r] > 0.01 and not in_row:
            in_row = True
            start_r = r
        elif row_density[r] <= 0.005 and in_row:
            in_row = False
            row_h = r - start_r
            if 15 <= row_h <= int(h * 0.12):
                sub = is_dark[start_r:r, :]
                col_proj = np.mean(sub, axis=0)
                active_cols = np.where(col_proj > 0.02)[0]
                if len(active_cols) > 0:
                    c_start, c_end = active_cols[0], active_cols[-1]
                    if (c_end - c_start) / w >= min_width_ratio:
                        ymin = max(0.0, float(start_r) / h)
                        xmin = max(0.0, float(c_start) / w)
                        ymax = min(1.0, float(r) / h)
                        xmax = min(1.0, float(c_end) / w)
                        rois.append((ymin, xmin, ymax, xmax))

    return rois


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
