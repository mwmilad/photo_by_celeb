"""Shared image helpers used by every module.

Conventions:
- Images are float32 NumPy arrays in 0..1, shape (H, W, 3) for RGB or (H, W, 4) for RGBA.
- Pixel values are sRGB (gamma encoded) unless a name says `lin` (linear light).
- Regions are normalized rectangles [x, y, w, h] in 0..1 of the image size,
  so the config keeps working when the camera resolution changes.
"""
import io

import numpy as np
from PIL import Image, ImageCms, ImageOps

# ---------------------------------------------------------------- color space

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_EPS = (6 / 29) ** 3


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin):
    """Relative luminance of linear RGB, shape (...,)."""
    return lin @ LUMA


def linear_to_lab(lin):
    xyz = (lin @ _RGB2XYZ.T) / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(np.maximum(xyz, 0)), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1).astype(np.float32)


def lab_to_linear(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    return np.clip(xyz @ _XYZ2RGB.T, 0, 1).astype(np.float32)


def rgb_to_lab(rgb):
    return linear_to_lab(srgb_to_linear(rgb))


def lab_to_rgb(lab):
    return linear_to_srgb(lab_to_linear(lab))


def color_cast(lin_color):
    """How far a (supposedly neutral) linear color is from gray, in Lab a/b units."""
    lab = linear_to_lab(np.asarray(lin_color, dtype=np.float32))
    return float(np.hypot(lab[..., 1], lab[..., 2]))


def to_255(rgb):
    return [int(round(float(v) * 255)) for v in np.clip(rgb, 0, 1)]


# ---------------------------------------------------------------- blur

def _box_blur_axis(a, r, axis):
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi, lo = [slice(None)] * a.ndim, [slice(None)] * a.ndim
    hi[axis], lo[axis] = slice(2 * r + 1, 2 * r + 1 + n), slice(0, n)
    return ((c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)).astype(np.float32)


def gaussian_blur(a, sigma):
    """Gaussian blur over axes 0 and 1 (OpenCV when installed, else 3 box passes)."""
    if sigma <= 0:
        return a.astype(np.float32, copy=True)
    a = np.ascontiguousarray(a, dtype=np.float32)
    try:
        import cv2
        if a.ndim == 2 or a.shape[-1] <= 4:
            return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE).reshape(a.shape)
    except ImportError:
        pass
    if sigma <= 3:  # small blur: exact separable kernel
        r = int(np.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
        k /= k.sum()
        out = a
        for axis in (0, 1):
            pad = [(0, 0)] * a.ndim
            pad[axis] = (r, r)
            p = np.pad(out, pad, mode="edge")
            n = out.shape[axis]
            acc = np.zeros_like(out)
            for i, w in enumerate(k):
                sl = [slice(None)] * a.ndim
                sl[axis] = slice(i, i + n)
                acc += w * p[tuple(sl)]
            out = acc
        return out
    r = max(1, int(round((np.sqrt(12 * sigma ** 2 / 3 + 1) - 1) / 2)))
    out = a
    for _ in range(3):
        out = _box_blur_axis(_box_blur_axis(out, r, 0), r, 1)
    return out


def blur_premultiplied(rgba, sigma):
    """Blur the colors of an RGBA layer without dark fringes; alpha is kept."""
    a = rgba[..., 3:4]
    num = gaussian_blur(rgba[..., :3] * a, sigma)
    den = gaussian_blur(a, sigma)
    rgb = np.where(den > 1e-4, num / np.maximum(den, 1e-4), rgba[..., :3])
    return np.concatenate([np.clip(rgb, 0, 1), a], axis=-1)


# ---------------------------------------------------------------- regions

def region_slices(shape, rect):
    """Normalized [x, y, w, h] -> (row slice, col slice), always at least 1 px."""
    h, w = shape[:2]
    x, y, rw, rh = rect
    x0, y0 = int(np.clip(round(x * w), 0, w - 1)), int(np.clip(round(y * h), 0, h - 1))
    x1, y1 = int(np.clip(round((x + rw) * w), x0 + 1, w)), int(np.clip(round((y + rh) * h), y0 + 1, h))
    return slice(y0, y1), slice(x0, x1)


def region_pixels(img, rect, mask=None):
    ys, xs = region_slices(img.shape, rect)
    px = img[ys, xs]
    if mask is not None:
        px = px[mask[ys, xs]]
    return px.reshape(-1, img.shape[-1])


def robust_color(pixels):
    """Median color, ignoring clipped pixels (they lie about the light)."""
    if len(pixels) == 0:
        return None
    ok = pixels.max(axis=1) < 0.98
    if ok.sum() >= max(20, 0.2 * len(pixels)):
        pixels = pixels[ok]
    return np.median(pixels, axis=0).astype(np.float32)


def bbox(mask):
    """(y0, y1, x0, x1) of True pixels, or None."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1) + 1, int(x0), int(x1) + 1


def padded_bbox(mask, margin):
    """bbox() grown by `margin` pixels and clipped to the image, as a (rows, cols) slice pair."""
    b = bbox(mask)
    if b is None:
        return None
    h, w = mask.shape[:2]
    y0, y1, x0, x1 = b
    return slice(max(y0 - margin, 0), min(y1 + margin, h)), slice(max(x0 - margin, 0), min(x1 + margin, w))


# ---------------------------------------------------------------- PIL <-> NumPy

def from_pil(img):
    return np.asarray(img, dtype=np.float32) / 255


def to_pil(arr):
    arr = np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def normalize_pil(img, mode):
    """Apply EXIF rotation and convert any embedded color profile to sRGB."""
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    converted = False
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            base = img.convert("RGB")
            rgb = ImageCms.profileToProfile(base, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
            if mode == "RGBA":
                rgb.putalpha(img.convert("RGBA").getchannel("A"))
            img, converted = rgb, True
        except Exception:
            pass  # unreadable profile: treat as sRGB
    return img.convert(mode), converted


# ---------------------------------------------------------------- faces (optional OpenCV)

_cascades = {}


def _cascade(name):
    try:
        import cv2
    except ImportError:
        return None
    if name not in _cascades:
        _cascades[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _cascades[name]


def detect_face(rgb, background=0.5):
    """Largest frontal face as dict(x, y, w, h) in pixels; None if none found.

    Returns False if OpenCV is not installed (so callers can tell "no detector"
    from "no face"). RGBA input is flattened on gray first.
    """
    face_cascade = _cascade("haarcascade_frontalface_default.xml")
    if face_cascade is None:
        return False
    import cv2
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3] * rgb[..., 3:] + background * (1 - rgb[..., 3:])
    gray = np.round(np.clip(luminance(rgb), 0, 1) * 255).astype(np.uint8)
    min_side = max(24, min(gray.shape) // 40)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "eye_y": int(y + 0.4 * h), "cx": int(x + w / 2)}


def count_open_eyes(rgb, face):
    """Number of eyes found in the upper half of a face (None without OpenCV)."""
    eye_cascade = _cascade("haarcascade_eye.xml")
    if eye_cascade is None or not face:
        return None
    gray = np.round(np.clip(luminance(rgb), 0, 1) * 255).astype(np.uint8)
    roi = gray[face["y"]:face["y"] + face["h"] // 2 + face["h"] // 8, face["x"]:face["x"] + face["w"]]
    if roi.size == 0:
        return 0
    side = max(8, face["w"] // 10)
    return int(len(eye_cascade.detectMultiScale(roi, 1.1, 4, minSize=(side, side))))
