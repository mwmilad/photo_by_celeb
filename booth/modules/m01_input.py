"""Input: same orientation, size and color space for every photo.

Rotates/mirrors the camera photo (and the user mask), converts embedded color
profiles to sRGB and resizes to a fixed working height.
"""
from PIL import Image, ImageOps

from ..imaging import from_pil, normalize_pil

KEY = "input"
TITLE = "Input normalization"
REQUIRED = True

_ROTATIONS = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def _orient(img, cfg, size):
    if cfg["rotate"] % 360:
        img = img.transpose(_ROTATIONS[cfg["rotate"] % 360])
    if cfg["mirror"]:
        img = ImageOps.mirror(img)
    return img.resize(size, Image.Resampling.LANCZOS) if size and img.size != size else img


def run(ctx, config):
    cfg = config[KEY]
    if cfg["rotate"] % 360 not in (0, 90, 180, 270):
        raise ValueError("input.rotate must be 0, 90, 180 or 270")

    camera, camera_icc = normalize_pil(ctx.raw_camera, "RGB")
    camera = _orient(camera, cfg, None)
    if cfg["working_height"] and camera.height != cfg["working_height"]:
        w = round(camera.width * cfg["working_height"] / camera.height)
        camera = camera.resize((w, cfg["working_height"]), Image.Resampling.LANCZOS)
    ctx.camera = from_pil(camera)

    celeb, celeb_icc = normalize_pil(ctx.raw_celeb, "RGBA")
    ctx.celeb = from_pil(celeb)

    if ctx.raw_user_mask is not None:
        mask = ImageOps.exif_transpose(ctx.raw_user_mask).convert("L")
        ctx.user_mask = from_pil(_orient(mask, cfg, camera.size))

    if ctx.celeb[..., 3].min() > 0.99:
        ctx.warn(KEY, "celeb frame has no transparency: is it really RGBA?")
    return {
        "camera_size": list(camera.size),
        "camera_raw_size": list(ctx.raw_camera.size),
        "celeb_size": list(celeb.size),
        "color_profile_converted": {"camera": camera_icc, "celeb": celeb_icc},
        "user_mask": ctx.user_mask is not None,
    }
