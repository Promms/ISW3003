import os
import numpy as np
from PIL import Image

# Pascal VOC 2012 segmentation class <-> palette color (bit-shift generation).
# index  0: background    (  0,   0,   0)
# index  1: aeroplane     (128,   0,   0)
# index  2: bicycle       (  0, 128,   0)
# index  3: bird          (128, 128,   0)
# index  4: boat          (  0,   0, 128)
# index  5: bottle        (128,   0, 128)
# index  6: bus           (  0, 128, 128)
# index  7: car           (128, 128, 128)
# index  8: cat           ( 64,   0,   0)
# index  9: chair         (192,   0,   0)
# index 10: cow           ( 64, 128,   0)
# index 11: diningtable   (192, 128,   0)
# index 12: dog           ( 64,   0, 128)
# index 13: horse         (192,   0, 128)
# index 14: motorbike     ( 64, 128, 128)
# index 15: person        (192, 128, 128)
# index 16: pottedplant   (  0,  64,   0)
# index 17: sheep         (128,  64,   0)
# index 18: sofa          (  0, 192,   0)
# index 19: train         (128, 192,   0)
# index 20: tvmonitor     (  0,  64, 128)
# index 255: void/ignore  (224, 224, 192)  # unlabeled boundary pixels

def voc_palette(n: int = 256) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


VOC_PALETTE = voc_palette()

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def colorize_mask(mask_arr: np.ndarray) -> np.ndarray:
    """Map a class-index mask (H, W) to an RGB image (H, W, 3) using VOC_PALETTE."""
    return VOC_PALETTE[mask_arr]


def overlay(img_arr: np.ndarray, mask_arr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the colorized mask on top of the image, keeping background pixels untouched."""
    color = colorize_mask(mask_arr).astype(np.float32)
    base = img_arr.astype(np.float32)
    blended = base * (1 - alpha) + color * alpha
    fg = mask_arr != 0
    out = base.copy()
    out[fg] = blended[fg]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_panel(img_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """Build a horizontal [image | colorized mask | overlay] visualization panel."""
    mask_rgb = colorize_mask(mask_arr)
    ov = overlay(img_arr, mask_arr)
    return np.concatenate([img_arr, mask_rgb, ov], axis=1)


def load_pair(root: str, name: str):
    """Load (img, mask) from `{root}/img/{name}.jpg` and `{root}/mask/{name}.png`.

    The mask is forced to L mode so VOC class indices are preserved as uint8.
    """
    img = Image.open(os.path.join(root, "img", f"{name}.jpg")).convert("RGB")
    mask = Image.open(os.path.join(root, "mask", f"{name}.png"))
    if mask.mode != "L":
        mask = mask.convert("L")
    return img, mask
