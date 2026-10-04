"""Color match: the celeb looks lit by the same light as the room.

method "neutral" (recommended) matches the LIGHT, not the content: it compares
something neutral on the celeb (white shirt, gray card, or auto-found grayish
pixels) with something neutral in the scene (reference card / wall) and only
changes the light color. Skin tones and clothes keep their own colors, so a
pale or dark-skinned user never changes your friend's skin.

method "stats" is the old whole-image match (match_colors.py) for comparison.
"""
import numpy as np

from ..imaging import (color_cast, lab_to_linear, linear_to_lab, linear_to_srgb, luminance,
                       region_pixels, robust_color, srgb_to_linear, to_255)

KEY = "color_match"
TITLE = "Color match (celeb → scene)"
REQUIRED = False


def _neutral_candidates(lin, mask, max_chroma=25):
    """Pixels that are probably white/gray objects lit by the scene light.

    Allows a strong color cast (that's what we want to measure), but skips
    skin-like hues and keeps the brightest candidates: a white shirt or wall
    is a better light reference than dark, noisy pixels.
    """
    px = lin[mask]
    if len(px) > 200_000:
        px = px[:: len(px) // 200_000]
    lab = linear_to_lab(px)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    skin = (hue > 15) & (hue < 75) & (chroma > 6)
    keep = (chroma < max_chroma) & ~skin & (lab[:, 0] > 30) & (lab[:, 0] < 97)
    px, light = px[keep], lab[keep, 0]
    if len(px) < 300:
        return px
    return px[light >= np.percentile(light, 75)]


def _scene_neutral(ctx, config, lin):
    scene, choice = config["scene"], config[KEY]["scene_region"]
    for name in (["reference_patch", "wall"] if choice == "auto" else
                 ["reference_patch" if choice == "patch" else "wall"]):
        if scene.get(name):
            return robust_color(region_pixels(lin, scene[name])), name
    mask = np.ones(lin.shape[:2], bool)
    if ctx.celeb_layer is not None:
        mask &= ctx.celeb_layer[..., 3] < 0.01
    px = _neutral_candidates(lin, mask)
    return (np.median(px, axis=0), "auto_gray_pixels") if len(px) > 75 else (None, None)


def _celeb_neutral(ctx, lin):
    solid = ctx.celeb_layer[..., 3] > 0.95
    rect = ctx.celeb_info.get("neutral_region")
    if rect:
        y0, y1, x0, x1 = ctx.celeb_rect_to_canvas(rect)
        box = np.zeros_like(solid)
        box[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        px = lin[solid & box]
        return (robust_color(px), "config_region") if len(px) > 20 else (None, None)
    f = ctx.celeb_face
    if f:  # never use the face or hair as a color reference
        solid = solid.copy()
        solid[max(f["y"] - f["h"], 0):f["y"] + int(f["h"] * 1.4), max(f["x"] - f["w"] // 2, 0):f["x"] + int(f["w"] * 1.5)] = False
    px = _neutral_candidates(lin, solid)
    return (np.median(px, axis=0), "auto_gray_pixels") if len(px) > 75 else (None, None)


def _clamp_shift(gains, max_shift):
    """Limit how much a mid-gray changes color, in Lab a/b units."""
    gray = np.full(3, 0.18, np.float32)
    shift = color_cast(gray * gains)
    if shift <= max_shift or shift == 0:
        return gains, False
    return gains ** (max_shift / shift), True


def _stats(ctx, config, lin_layer, solid):
    """Old content-based match: mean/std of celeb pixels -> camera pixels (Reinhard, Lab)."""
    cam = linear_to_lab(srgb_to_linear(ctx.camera[::4, ::4]).reshape(-1, 3))
    src = linear_to_lab(lin_layer[solid])
    out = (src - src.mean(0)) / (src.std(0) + 1e-6) * cam.std(0) + cam.mean(0)
    s = config[KEY]["strength"]
    lin_layer[solid] = lab_to_linear(src + (out - src) * s)
    return {"method": "stats"}


def run(ctx, config):
    cfg = config[KEY]
    layer = ctx.celeb_layer
    lin_layer = srgb_to_linear(layer[..., :3])
    visible = layer[..., 3] > 0

    if cfg["method"] == "stats":
        m = _stats(ctx, config, lin_layer, visible)
        ctx.celeb_layer = np.concatenate([linear_to_srgb(lin_layer), layer[..., 3:]], axis=-1)
        return m

    scene, scene_src = _scene_neutral(ctx, config, srgb_to_linear(ctx.camera))
    celeb, celeb_src = _celeb_neutral(ctx, lin_layer)
    if scene is None or celeb is None:
        ctx.warn(KEY, f"could not measure neutral color on the {'scene' if scene is None else 'celeb'}: skipped")
        return {"method": "neutral", "skipped": True}

    # light color only: same brightness, different tint
    gains = (scene / luminance(scene)) / np.maximum(celeb / luminance(celeb), 1e-6)
    if cfg["brightness"] == "neutral":
        gains = gains * luminance(scene) / luminance(celeb)
    elif cfg["brightness"] == "manual":
        gains = gains * cfg["brightness_gain"]
    gains = gains ** cfg["strength"]
    gains, limited = _clamp_shift(gains.astype(np.float32), cfg["max_shift"])
    if limited:
        ctx.warn(KEY, "color change was limited by max_shift: check the neutral regions")

    lin_layer[visible] *= gains
    ctx.celeb_layer = np.concatenate([linear_to_srgb(lin_layer), layer[..., 3:]], axis=-1)

    def delta(a, b):
        la, lb = linear_to_lab(a / luminance(a) * 0.18), linear_to_lab(b / luminance(b) * 0.18)
        return round(float(np.hypot(*(la[1:] - lb[1:]))), 2)

    return {
        "method": "neutral",
        "scene_neutral": to_255(linear_to_srgb(scene)), "scene_source": scene_src,
        "celeb_neutral": to_255(linear_to_srgb(celeb)), "celeb_source": celeb_src,
        "gains": [round(float(g), 3) for g in gains],
        "color_delta_before": delta(celeb, scene),
        "color_delta_after": delta(celeb * gains, scene),
        "limited": limited,
    }
