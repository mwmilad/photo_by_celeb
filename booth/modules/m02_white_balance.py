"""White balance: make neutral things in the camera photo really neutral.

Methods:
- patch:       measure the white/gray reference card (most reliable: clothes can't fool it)
- gray_pixels: guess from pixels that are already almost gray (no card needed, less reliable)
- calibration: fixed gains measured once per event
"""
import numpy as np

from ..imaging import (color_cast, linear_to_lab, linear_to_srgb, luminance, region_pixels,
                       robust_color, srgb_to_linear, to_255)

KEY = "white_balance"
TITLE = "White balance"
REQUIRED = False


def _gray_pixels_neutral(lin):
    small = lin[::4, ::4].reshape(-1, 3)
    lab = linear_to_lab(small)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (chroma < 8) & (lab[:, 0] > 15) & (lab[:, 0] < 95)
    if keep.mean() < 0.005:  # almost nothing gray: fall back to gray world
        return np.median(small, axis=0), "gray_world"
    return np.median(small[keep], axis=0), "gray_pixels"


def run(ctx, config):
    cfg, patch = config[KEY], config["scene"]["reference_patch"]
    lin = srgb_to_linear(ctx.camera)
    method = cfg["method"]
    if method == "patch" and not patch:
        ctx.warn(KEY, "no scene.reference_patch configured: using gray_pixels")
        method = "gray_pixels"

    if method == "calibration":
        gains, neutral = np.array(cfg["gains"], dtype=np.float32), None
    else:
        if method == "patch":
            neutral = robust_color(region_pixels(lin, patch))
        else:
            neutral, method = _gray_pixels_neutral(lin)
        gains = luminance(neutral) / np.maximum(neutral, 1e-6)  # keep brightness, remove color

    gains = np.clip(gains, 1 / cfg["max_gain"], cfg["max_gain"]) ** cfg["strength"]
    ctx.camera = linear_to_srgb(lin * gains)

    m = {"method": method, "gains": [round(float(g), 3) for g in gains]}
    if neutral is not None:
        m["neutral_before"] = to_255(linear_to_srgb(neutral))
        m["cast_before"] = round(color_cast(neutral), 2)
        m["cast_after"] = round(color_cast(neutral * gains), 2)
    return m
