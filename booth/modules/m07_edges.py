"""Edges: no halos, fringes or cut-out look.

1. shrink the matte a little (removes the old background's outline)
2. feather it (soft natural edge)
3. decontaminate: edge pixels take the color of nearby inside pixels
4. spill: remove green/blue screen color left in hair
5. light wrap: the room's light leaks slightly over the celeb's edges
"""
import numpy as np
from PIL import Image, ImageFilter

from ..imaging import gaussian_blur, linear_to_srgb, padded_bbox, srgb_to_linear

KEY = "edges"
TITLE = "Edges (matte, halo, light wrap)"
REQUIRED = False


def run(ctx, config):
    cfg = config[KEY]
    area = padded_bbox(ctx.celeb_layer[..., 3] > 0, int(3 * max(cfg["light_wrap_px"], 4)) + 2)
    if area is None:
        return {"skipped": "empty celeb layer"}
    layer, camera = ctx.celeb_layer[area], ctx.camera[area]  # work only around the celeb
    rgb, alpha = layer[..., :3].copy(), layer[..., 3].copy()
    edge_before = int(((alpha > 0.02) & (alpha < 0.98)).sum())

    if cfg["shrink_px"] > 0:
        a8 = Image.fromarray(np.round(alpha * 255).astype(np.uint8), "L")
        alpha = np.asarray(a8.filter(ImageFilter.MinFilter(2 * int(cfg["shrink_px"]) + 1)), np.float32) / 255
    if cfg["feather_px"] > 0:
        alpha = np.minimum(gaussian_blur(alpha, cfg["feather_px"]), layer[..., 3])

    edge = (alpha > 0) & (alpha < 0.98)
    if cfg["decontaminate"]:
        inside = (layer[..., 3] > 0.98).astype(np.float32)
        spread = gaussian_blur(rgb * inside[..., None], 4) / np.maximum(gaussian_blur(inside, 4), 1e-4)[..., None]
        has = gaussian_blur(inside, 4) > 0.02
        w = np.where(edge & has, 1 - alpha, 0)[..., None]
        rgb = rgb * (1 - w) + spread * w

    if cfg["spill"] in ("green", "blue"):
        c = 1 if cfg["spill"] == "green" else 2
        others = np.max(np.delete(rgb, c, axis=-1), axis=-1)
        near_edge = gaussian_blur(edge.astype(np.float32), 3) > 0.01
        rgb[..., c] = np.where(near_edge, np.minimum(rgb[..., c], others), rgb[..., c])

    if cfg["light_wrap"] > 0:
        bg = srgb_to_linear(gaussian_blur(camera, cfg["light_wrap_px"]))
        inner = gaussian_blur(alpha, cfg["light_wrap_px"] / 2)
        wrap = np.clip(alpha * (1 - inner) * 2, 0, 1) * cfg["light_wrap"]
        lin = srgb_to_linear(rgb)
        rgb = linear_to_srgb(lin + (bg - lin) * wrap[..., None])

    out = ctx.celeb_layer.copy()
    out[area] = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    ctx.celeb_layer = out
    return {"edge_pixels_before": edge_before, "edge_pixels_after": int(edge.sum()),
            "shrink_px": cfg["shrink_px"], "feather_px": cfg["feather_px"], "light_wrap": cfg["light_wrap"]}
