"""Runs the modules in order and (optionally) keeps a snapshot after each one."""
import os
import time

from PIL import Image

from .config import celeb_settings, load_config
from .context import Context
from .imaging import linear_to_srgb, srgb_to_linear, to_pil
from .modules import MODULES


def _open(img):
    """Path, file object (e.g. FastAPI UploadFile.file) or PIL image -> PIL image."""
    return img if isinstance(img, Image.Image) else Image.open(img)


def preview(ctx):
    """What the photo looks like right now, as a canvas-sized RGB array."""
    if ctx.final is not None:
        return ctx.final
    if ctx.composite is not None:
        return ctx.composite
    if ctx.camera is None:
        return None
    if ctx.celeb_layer is None:
        return ctx.camera
    a = ctx.celeb_layer[..., 3:4]
    return linear_to_srgb(srgb_to_linear(ctx.celeb_layer[..., :3]) * a + srgb_to_linear(ctx.camera) * (1 - a))


def run(camera, celeb, config=None, celeb_name=None, user_mask=None, disable=(), snapshots=False):
    """Make the final photo.

    camera, celeb, user_mask: file paths, file objects or PIL images.
    config: dict from load_config() or a path to a JSON file.
    disable: module keys to skip (required modules always run).
    Returns the Context (ctx.outputs has "print" and "share" PIL images) and,
    with snapshots=True, a list of dicts {key, title, image, metrics, skipped, ms}.
    """
    if not isinstance(config, dict):
        config = load_config(config)
    if celeb_name is None:
        celeb_name = os.path.splitext(os.path.basename(celeb))[0] if isinstance(celeb, (str, os.PathLike)) else "default"
    ctx = Context(raw_camera=_open(camera), raw_celeb=_open(celeb),
                  celeb_info=celeb_settings(config, celeb_name),
                  raw_user_mask=_open(user_mask) if user_mask is not None else None)

    shots = []
    for module in MODULES:
        cfg = config.get(module.KEY, {})
        skipped = not module.REQUIRED and (module.KEY in disable or not cfg.get("enabled", True))
        t0 = time.perf_counter()
        if not skipped:
            ctx.metrics[module.KEY] = module.run(ctx, config)
        ms = round((time.perf_counter() - t0) * 1000)
        if snapshots:
            if module.KEY in ("export", "quality") and "print" in ctx.outputs:
                img = ctx.outputs["print"]
            else:
                img = to_pil(preview(ctx))
            shots.append({"key": module.KEY, "title": module.TITLE, "image": img, "skipped": skipped,
                          "metrics": ctx.metrics.get(module.KEY, {}), "ms": ms,
                          "warnings": [w for k, w in ctx.warnings if k == module.KEY]})

    if not ctx.outputs:  # export disabled: still hand back something printable
        final = to_pil(ctx.final if ctx.final is not None else ctx.composite)
        ctx.outputs = {"print": final, "share": final}
    return (ctx, shots) if snapshots else ctx

