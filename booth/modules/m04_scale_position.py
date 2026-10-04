"""Scale & position: the celeb's head size and eye line fit the user.

The scale is decided once from the user's face and clamped to +-max_change of
the base size. Because the celeb's body is cut by the frame bottom, the bottom
edge is never lifted above the canvas edge (that would show a floating body).
With auto off (or no OpenCV / no face) the celeb is placed at the base size.
"""
import numpy as np
from PIL import Image

from ..imaging import detect_face, from_pil, to_pil

KEY = "scale_position"
TITLE = "Scale & position"
REQUIRED = True


def _celeb_face(ctx):
    info = ctx.celeb_info
    if info["head_height"] and info["eye_y"]:
        return {"h": info["head_height"], "eye_y": info["eye_y"], "source": "config"}
    face = detect_face(ctx.celeb)
    return dict(face, source="detected") if face else face


def run(ctx, config):
    cfg = config[KEY]
    H, W = ctx.camera.shape[:2]
    ch, cw = ctx.celeb.shape[:2]
    base = H * cfg["base_height"] / ch
    scale, y, m = base, None, {"base_scale": round(base, 4), "mode": "base"}

    user, celeb = detect_face(ctx.camera), _celeb_face(ctx)
    m["face_detector"] = user is not False
    if not m["face_detector"]:
        ctx.warn(KEY, "OpenCV not installed: no face detection, base size used "
                      "(pip install 'opencv-python-headless<5')")
        user = celeb = None
    ctx.user_face = user

    if cfg["auto"] and m["face_detector"]:
        if not user:
            ctx.warn(KEY, "no user face found: using base size")
        elif not celeb:
            ctx.warn(KEY, "no face found on celeb frame: set head_height/eye_y in the config")
        else:
            wanted = user["h"] * ctx.celeb_info["height_ratio"] / celeb["h"]
            adjust = float(np.clip(wanted / base, 1 - cfg["max_change"], 1 + cfg["max_change"]))
            scale = base * adjust
            y = user["eye_y"] + cfg["eye_offset"] * H - celeb["eye_y"] * scale
            m.update(mode="auto", wanted_adjust=round(wanted / base, 3), adjust=round(adjust, 3),
                     clamped=abs(wanted / base - adjust) > 1e-3, user_face=user,
                     celeb_face_source=celeb["source"])

    sw, sh = max(1, round(cw * scale)), max(1, round(ch * scale))
    bottom_y = H - sh
    if y is None:
        y = bottom_y
    elif ctx.celeb_info["cut_at_bottom"] and y < bottom_y:
        m["eye_line_limited"] = True  # moving up would show the cut body
        y = bottom_y
    x = W * cfg["x"] - sw / 2
    x, y = int(round(x)), int(round(y))

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(to_pil(ctx.celeb).resize((sw, sh), Image.Resampling.LANCZOS), (x, y))
    ctx.celeb_layer = from_pil(canvas)
    ctx.celeb_transform = {"scale": sw / cw, "x": x, "y": y}

    if celeb and "x" in celeb:  # celeb face box on the canvas, used by light_shape and texture
        s = sw / cw
        shift = {"x": x, "cx": x, "y": y, "eye_y": y}
        ctx.celeb_face = {k: int(round(celeb[k] * s + shift.get(k, 0))) for k in ("x", "y", "w", "h", "eye_y", "cx")}

    m.update(scale=round(sw / cw, 4), position=[x, y], size=[sw, sh])
    return m
