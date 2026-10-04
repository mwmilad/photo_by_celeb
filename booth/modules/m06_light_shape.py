"""Light shape: same light direction and contrast on the celeb as on the user.

auto: measures the left/right brightness balance of the user's face and of the
celeb's face, and adds a horizontal gradient to the celeb so both match.
manual: a fixed gradient from `direction_deg` (0 = light from left, 90 = top,
180 = right). `contrast` scales celeb contrast around its median lightness.
"""
import numpy as np

from ..imaging import bbox, lab_to_linear, linear_to_lab, linear_to_srgb, luminance, srgb_to_linear

KEY = "light_shape"
TITLE = "Light shape (direction & contrast)"
REQUIRED = False


def _side_ratio(lin, face, alpha=None):
    """Right half / left half brightness of a face box (log), or None."""
    y0, y1 = face["y"] + face["h"] // 6, face["y"] + face["h"] * 5 // 6
    x0, x1, xm = face["x"] + face["w"] // 8, face["x"] + face["w"] * 7 // 8, face["cx"]
    if y0 < 0 or x0 < 0 or y1 > lin.shape[0] or x1 > lin.shape[1] or xm - x0 < 2:
        return None
    lum = luminance(lin[y0:y1, x0:x1])
    if alpha is not None and alpha[y0:y1, x0:x1].min() < 0.9:
        return None
    left, right = np.median(lum[:, : xm - x0]), np.median(lum[:, xm - x0:])
    return float(np.log(max(right, 1e-4) / max(left, 1e-4)))


def run(ctx, config):
    cfg = config[KEY]
    layer = ctx.celeb_layer
    box = bbox(layer[..., 3] > 0.01)
    if box is None:
        return {"skipped": "empty celeb layer"}
    y0, y1, x0, x1 = box
    lin = srgb_to_linear(layer[..., :3])
    m = {}

    if cfg["auto"] and ctx.user_face and ctx.celeb_face:
        user = _side_ratio(srgb_to_linear(ctx.camera), ctx.user_face)
        celeb = _side_ratio(lin, ctx.celeb_face, layer[..., 3])
        if user is not None and celeb is not None:
            # log-brightness slope per pixel needed across the celeb's face, applied to the whole body
            slope = (user - celeb) / (ctx.celeb_face["w"] * 0.375)
            limit = np.log(1 + cfg["max_gradient"]) * 2 / max(x1 - x0, 1)
            slope = float(np.clip(slope, -limit, limit))
            xs = np.arange(x0, x1, dtype=np.float32) - ctx.celeb_face["cx"]
            gain = np.exp(slope * xs)[None, :, None]
            lin[y0:y1, x0:x1] *= gain
            m.update(mode="auto", user_balance=round(user, 3), celeb_balance_before=round(celeb, 3),
                     celeb_balance_after=round(_side_ratio(lin, ctx.celeb_face, layer[..., 3]) or 0, 3))
        else:
            ctx.warn(KEY, "faces not fully visible: auto light shape skipped")
    elif cfg["auto"]:
        ctx.warn(KEY, "no faces detected: using manual light shape")

    if "mode" not in m and cfg["gradient"]:
        a = np.radians(cfg["direction_deg"])
        lx, ly = -np.cos(a), -np.sin(a)  # points toward the light (image y grows downward)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        t = ((xx - (x0 + x1) / 2) / max(x1 - x0, 1) * lx + (yy - (y0 + y1) / 2) / max(y1 - y0, 1) * ly) * 2
        lin[y0:y1, x0:x1] *= (1 + cfg["gradient"] * np.clip(t, -1, 1))[..., None]
        m.update(mode="manual", direction_deg=cfg["direction_deg"], gradient=cfg["gradient"])

    if cfg["contrast"] != 1.0:
        solid = layer[..., 3] > 0.5
        lab = linear_to_lab(lin[solid])
        mid = np.median(lab[:, 0])
        lab[:, 0] = np.clip(mid + (lab[:, 0] - mid) * cfg["contrast"], 0, 100)
        lin[solid] = lab_to_linear(lab)
        m["contrast"] = cfg["contrast"]

    ctx.celeb_layer = np.concatenate([linear_to_srgb(lin), layer[..., 3:]], axis=-1)
    return m or {"mode": "none"}
