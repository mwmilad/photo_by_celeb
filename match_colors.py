"""Make RGBA images match the color, brightness and contrast of a reference image.

Only visible pixels are measured and changed; the alpha channel is kept as is.

Usage:
    python match_colors.py reference.png target.png -o fixed.png
    python match_colors.py reference.png assets/*.png -o fixed/      # many files
    python match_colors.py reference.png target.png -o fixed.png --method hist --strength 0.8
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ALPHA_THRESHOLD = 128  # pixels with alpha below this are ignored when measuring

# sRGB (D65) <-> XYZ
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    """rgb: float array (..., 3) in 0..1 -> Lab (L 0..100)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _RGB2XYZ.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = np.clip(xyz @ _XYZ2RGB.T, 0, 1)
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


def _load(path):
    img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255
    mask = img[..., 3] >= ALPHA_THRESHOLD / 255
    if not mask.any():
        raise ValueError(f"{path}: image is fully transparent")
    return img, mask


def _fit_mean_std(src, ref):
    """Reinhard transfer: same average and spread as the reference."""
    s_mean, s_std = src.mean(0), src.std(0) + 1e-6
    r_mean, r_std = ref.mean(0), ref.std(0)
    return lambda x: (x - s_mean) / s_std * r_std + r_mean


def _fit_histogram(src, ref):
    """Exact histogram matching per channel (stronger, can look harsher)."""
    q = np.linspace(0, 100, 257)
    s_q, r_q = np.percentile(src, q, axis=0), np.percentile(ref, q, axis=0)
    return lambda x: np.stack([np.interp(x[:, c], s_q[:, c], r_q[:, c])
                               for c in range(x.shape[1])], axis=1)


def match_colors(reference, target, method="mean", strength=1.0):
    """Return a new RGBA PIL image: `target` recolored to look like `reference`."""
    ref_img, ref_mask = _load(reference)
    tgt_img, tgt_mask = _load(target)

    # learn the correction from clearly visible pixels only
    fit = _fit_histogram if method == "hist" else _fit_mean_std
    transform = fit(rgb_to_lab(tgt_img[..., :3][tgt_mask]),
                    rgb_to_lab(ref_img[..., :3][ref_mask]))

    # apply it to every pixel that is at least a bit visible (incl. soft edges)
    visible = tgt_img[..., 3] > 0
    lab = rgb_to_lab(tgt_img[..., :3][visible])
    new_lab = lab + (transform(lab) - lab) * strength
    new_lab[:, 0] = np.clip(new_lab[:, 0], 0, 100)

    out = tgt_img.copy()
    out[..., :3][visible] = lab_to_rgb(new_lab)
    return Image.fromarray(np.round(out * 255).astype(np.uint8), "RGBA")


def describe(path):
    img, mask = _load(path)
    lab = rgb_to_lab(img[..., :3][mask])
    m, s = lab.mean(0), lab.std(0)
    return (f"brightness(L) {m[0]:5.1f}  contrast(L std) {s[0]:5.1f}  "
            f"green-red(a) {m[1]:+5.1f}  blue-yellow(b) {m[2]:+5.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", help="RGBA image whose look you like")
    p.add_argument("targets", nargs="+", help="RGBA image(s) to fix")
    p.add_argument("-o", "--output", required=True,
                   help="output file (one target) or folder (several targets)")
    p.add_argument("--method", choices=["mean", "hist"], default="mean",
                   help="mean = natural (default), hist = stronger, exact match")
    p.add_argument("--strength", type=float, default=1.0,
                   help="0 = no change, 1 = full match (default 1.0)")
    args = p.parse_args()

    many = len(args.targets) > 1 or os.path.isdir(args.output)
    if many:
        os.makedirs(args.output, exist_ok=True)

    print(f"reference  {describe(args.reference)}  ({args.reference})")
    for target in args.targets:
        out_path = os.path.join(args.output, os.path.basename(target)) if many else args.output
        try:
            match_colors(args.reference, target, args.method, args.strength).save(out_path)
        except Exception as e:
            print(f"skipped {target}: {e}", file=sys.stderr)
            continue
        print(f"before     {describe(target)}  ({target})")
        print(f"after      {describe(out_path)}  ({out_path})")


if __name__ == "__main__":
    main()
