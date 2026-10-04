"""Composite: put the celeb over the camera photo.

Blending in linear light gives correct soft edges. Blending sRGB values
directly ("gamma", how most simple code does it) makes semi-transparent edges
too dark: the thin dark outline around cut-outs. With a user mask the user is
drawn back on top, so an arm in front of the celeb stays in front.
"""
from ..imaging import linear_to_srgb, srgb_to_linear

KEY = "composite"
TITLE = "Composite (linear blend)"
REQUIRED = True


def run(ctx, config):
    cfg = config[KEY]
    a = ctx.celeb_layer[..., 3:4]
    if ctx.user_mask is not None and cfg["user_in_front"]:
        a = a * (1 - ctx.user_mask[..., None])
    if cfg["space"] == "gamma":
        ctx.composite = ctx.celeb_layer[..., :3] * a + ctx.camera * (1 - a)
    else:
        lin = srgb_to_linear(ctx.celeb_layer[..., :3]) * a + srgb_to_linear(ctx.camera) * (1 - a)
        ctx.composite = linear_to_srgb(lin)
    return {"space": cfg["space"], "user_in_front": ctx.user_mask is not None and cfg["user_in_front"]}
