"""Export: a print version and a share version.

Print: crop to the paper size, print boost (paper is darker than screens),
sharpening, optional printer ICC profile, frame and AI label.
Share: smaller JPEG without the print boost (for the QR link).
"""
from PIL import Image, ImageCms, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ..imaging import to_pil

KEY = "export"
TITLE = "Print & share export"
REQUIRED = False


def crop_to_aspect(img, aspect, anchor):
    w, h = img.size
    if w / h > aspect:
        nw = round(h * aspect)
        x = round((w - nw) * anchor[0])
        return img.crop((x, 0, x + nw, h))
    nh = round(w / aspect)
    y = round((h - nh) * anchor[1])
    return img.crop((0, y, w, y + nh))


def _label(img, text):
    draw = ImageDraw.Draw(img)
    size = max(12, img.height // 90)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    x, y = img.width - (x1 - x0) - size, img.height - (y1 - y0) - size
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def run(ctx, config):
    cfg = config[KEY]
    final = to_pil(ctx.final if ctx.final is not None else ctx.composite)
    pw, ph = cfg["print_size_in"]
    if (final.width > final.height) != (pw > ph):
        pw, ph = ph, pw  # follow the photo's orientation
    size = (round(pw * cfg["dpi"]), round(ph * cfg["dpi"]))

    cropped = crop_to_aspect(final, pw / ph, cfg["crop_anchor"])
    if cropped.width < size[0] * 0.75:
        ctx.warn(KEY, f"photo is only {cropped.width}px wide for a {size[0]}px print: it will look soft")

    share = cropped.copy()
    share.thumbnail((cfg["share_max_size"],) * 2, Image.Resampling.LANCZOS)

    printed = cropped.resize(size, Image.Resampling.LANCZOS)
    printed = ImageEnhance.Brightness(printed).enhance(cfg["print_brightness"])
    printed = ImageEnhance.Color(printed).enhance(cfg["print_saturation"])
    if cfg["sharpen"]:
        printed = printed.filter(ImageFilter.UnsharpMask(radius=2, percent=int(cfg["sharpen"] * 100), threshold=2))
    if cfg["frame"]:
        frame = Image.open(cfg["frame"]).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        printed = Image.alpha_composite(printed.convert("RGBA"), frame).convert("RGB")
    if cfg["label"]:
        _label(printed, cfg["label"])
        _label(share, cfg["label"])
    if cfg["icc_profile"]:
        try:
            printed = ImageCms.profileToProfile(printed, ImageCms.createProfile("sRGB"), cfg["icc_profile"],
                                                outputMode="RGB")
        except (OSError, ImageCms.PyCMSError) as e:  # missing file or a CMYK profile
            ctx.warn(KEY, f"ICC profile not applied ({e}): printing plain sRGB")

    ctx.outputs = {"print": printed, "share": share}
    return {"print_px": list(printed.size), "dpi": cfg["dpi"], "share_px": list(share.size),
            "icc_profile": cfg["icc_profile"]}


def save_outputs(ctx, config, folder, stem):
    """Write print PNG (with DPI) and share JPEG; returns their paths."""
    cfg = config[KEY]
    paths = {}
    if "print" in ctx.outputs:
        paths["print"] = f"{folder}/{stem}_print.png"
        ctx.outputs["print"].save(paths["print"], dpi=(cfg["dpi"], cfg["dpi"]))
    if "share" in ctx.outputs:
        paths["share"] = f"{folder}/{stem}_share.jpg"
        ctx.outputs["share"].save(paths["share"], quality=cfg["jpeg_quality"], optimize=True)
    return paths

