"""Exposure: the room has the same brightness in every photo.

Measures a fixed area (reference card or wall, never the user) and scales the
light so it hits `target`. With target = None it only measures: run it once on
the empty booth and copy the measured value into the config.
"""
import numpy as np

from ..imaging import linear_to_srgb, luminance, region_pixels, robust_color, srgb_to_linear

KEY = "exposure"
TITLE = "Exposure"
REQUIRED = False


def run(ctx, config):
    cfg = config[KEY]
    rect = config["scene"]["reference_patch" if cfg["reference"] == "patch" else "wall"]
    lin = srgb_to_linear(ctx.camera)
    clipped = float((ctx.camera.max(axis=-1) > 0.99).mean())
    if not rect:
        ctx.warn(KEY, f"no scene region for reference '{cfg['reference']}': exposure unchanged")
        return {"clipped_fraction": round(clipped, 4)}

    level = float(luminance(robust_color(region_pixels(lin, rect))))
    measured = float(linear_to_srgb(np.float32(level)))
    m = {"reference": cfg["reference"], "measured": round(measured, 3), "clipped_fraction": round(clipped, 4)}
    if cfg["target"] is None:
        ctx.warn(KEY, f"exposure.target not set; this booth measures {measured:.3f}")
        return m

    target_lin = float(srgb_to_linear(np.float32(cfg["target"])))
    gain = float(np.clip(target_lin / max(level, 1e-6), 1 / cfg["max_gain"], cfg["max_gain"]))
    ctx.camera = linear_to_srgb(lin * gain)
    m.update(gain=round(gain, 3), after=round(float(linear_to_srgb(np.float32(level * gain))), 3))
    return m
