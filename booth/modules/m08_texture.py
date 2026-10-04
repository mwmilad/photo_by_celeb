"""Texture: the celeb gets the same softness and sensor noise as the camera.

blur "auto": compares fine detail of the user's face and the celeb's face (same
kind of content, so the comparison is fair) and softens the celeb until they match.
noise "auto": measures camera noise on the plain wall and adds only the missing part.
"""
import numpy as np
from PIL import Image

from ..imaging import blur_premultiplied, gaussian_blur, luminance, padded_bbox, region_slices

KEY = "texture"
TITLE = "Texture (sharpness & grain)"
REQUIRED = False


def _noise_sigma(rgb, mask=None):
    hp = rgb - gaussian_blur(rgb, 1.5)
    px = hp[mask] if mask is not None else hp.reshape(-1, 3)
    if len(px) < 50:
        return None
    return float(1.4826 * np.median(np.abs(px - np.median(px, axis=0))))


def _detail(rgb, face, size=128):
    crop = rgb[max(face["y"], 0):face["y"] + face["h"], max(face["x"], 0):face["x"] + face["w"]]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    img = Image.fromarray(np.round(np.clip(luminance(crop), 0, 1) * 255).astype(np.uint8), "L")
    g = np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.mean(np.abs(lap)))


def _auto_blur(ctx, max_blur):
    if not (ctx.user_face and ctx.celeb_face):
        return 0.0, "no faces"
    target = _detail(ctx.camera, ctx.user_face)
    f = ctx.celeb_face
    pad = 12
    y0, x0 = max(f["y"] - pad, 0), max(f["x"] - pad, 0)
    patch = ctx.celeb_layer[y0:f["y"] + f["h"] + pad, x0:f["x"] + f["w"] + pad]
    local = dict(f, y=f["y"] - y0, x=f["x"] - x0)
    current = _detail(patch[..., :3], local)
    if not target or not current or current <= target * 1.1:
        return 0.0, "celeb not sharper"
    best, best_err = 0.0, abs(np.log(current / target))
    for sigma in np.arange(0.3, max_blur + 0.01, 0.1):
        d = _detail(blur_premultiplied(patch, sigma)[..., :3], local)
        err = abs(np.log(d / target))
        if err < best_err:
            best, best_err = float(sigma), err
    return best, "matched faces"


def run(ctx, config):
    cfg = config[KEY]
    layer = ctx.celeb_layer
    m = {}

    sigma, why = (_auto_blur(ctx, cfg["max_blur"]) if cfg["blur"] == "auto" else (float(cfg["blur"] or 0), "manual"))
    m.update(blur_sigma=round(sigma, 2), blur_reason=why)
    area = padded_bbox(layer[..., 3] > 0, int(3 * sigma) + 2)
    if area is None:
        return m
    full, layer = layer, layer[area].copy()  # work only around the celeb
    if sigma > 0:
        layer = blur_premultiplied(layer, sigma)

    solid = layer[..., 3] > 0.98
    if cfg["noise"] == "auto":
        wall = config["scene"]["wall"]
        cam = _noise_sigma(ctx.camera[region_slices(ctx.camera.shape, wall)]) if wall else None
        if cam is None:
            ctx.warn(KEY, "no scene.wall configured: cannot measure camera noise")
            cam = 0.0
        own = _noise_sigma(layer[..., :3], solid) or 0.0
        add = float(np.sqrt(max(cam ** 2 - own ** 2, 0)))
        m.update(camera_noise=round(cam, 4), celeb_noise=round(own, 4))
    else:
        add = float(cfg["noise"] or 0)

    if add > 0:
        rng = np.random.default_rng(cfg["seed"])
        h, w = layer.shape[:2]
        mono = rng.normal(0, 1, (h, w, 1)).astype(np.float32)
        color = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
        k = cfg["noise_mono"]
        noise = (np.sqrt(k) * mono + np.sqrt(1 - k) * color) * add
        rgb = np.where(layer[..., 3:] > 0, np.clip(layer[..., :3] + noise, 0, 1), layer[..., :3])
        layer = np.concatenate([rgb, layer[..., 3:]], axis=-1)
    m["noise_added"] = round(add, 4)

    full = full.copy()
    full[area] = layer
    ctx.celeb_layer = full.astype(np.float32)
    return m
