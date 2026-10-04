"""Final grade: one shared look over the whole photo (the "glue").

The same LUT / saturation / contrast / vignette / grain on both people makes
small remaining differences disappear: they now share one photograph.
"""
import numpy as np

from ..imaging import lab_to_rgb, rgb_to_lab

KEY = "final_grade"
TITLE = "Final grade (shared look)"
REQUIRED = False


def load_cube(path):
    size, lo, hi, rows = None, np.zeros(3), np.ones(3), []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#") or parts[0] == "TITLE":
                continue
            if parts[0] == "LUT_3D_SIZE":
                size = int(parts[1])
            elif parts[0] == "DOMAIN_MIN":
                lo = np.array(parts[1:4], float)
            elif parts[0] == "DOMAIN_MAX":
                hi = np.array(parts[1:4], float)
            elif parts[0][0].isdigit() or parts[0][0] in "-.":
                rows.append([float(v) for v in parts[:3]])
    if not size or len(rows) != size ** 3:
        raise ValueError(f"{path}: not a valid 3D .cube LUT")
    # .cube order: red changes fastest -> index as [b, g, r]
    return np.array(rows, np.float32).reshape(size, size, size, 3), lo, hi


def apply_lut(rgb, lut, lo, hi):
    n = lut.shape[0]
    p = np.clip((rgb - lo) / (hi - lo), 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(p).astype(int), n - 2)
    f = p - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0
    for db, wb in ((0, 1 - fb), (1, fb)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            for dr, wr in ((0, 1 - fr), (1, fr)):
                out = out + lut[b0 + db, g0 + dg, r0 + dr] * (wb * wg * wr)
    return out.astype(np.float32)


def run(ctx, config):
    cfg = config[KEY]
    img = ctx.composite.copy()
    m = {}

    if cfg["lut"]:
        graded = apply_lut(img, *load_cube(cfg["lut"]))
        img = img + (graded - img) * cfg["lut_strength"]
        m["lut"] = cfg["lut"]

    if cfg["contrast"] != 1:  # gentle S-curve around mid-gray
        s = img * img * (3 - 2 * img)
        img = img + (s - img) * (cfg["contrast"] - 1) * 2
    if cfg["saturation"] != 1 or cfg["warmth"]:
        lab = rgb_to_lab(np.clip(img, 0, 1))
        lab[..., 1:] *= cfg["saturation"]
        lab[..., 2] += cfg["warmth"]
        img = lab_to_rgb(lab)

    h, w = img.shape[:2]
    if cfg["vignette"]:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.5
        img = img * (1 - cfg["vignette"] * d ** 1.5)[..., None]
    if cfg["grain"]:
        rng = np.random.default_rng(cfg["seed"])
        img = img + rng.normal(0, cfg["grain"], (h, w, 1)).astype(np.float32)

    ctx.final = np.clip(img, 0, 1).astype(np.float32)
    m.update({k: cfg[k] for k in ("saturation", "contrast", "warmth", "vignette", "grain")})
    return m
