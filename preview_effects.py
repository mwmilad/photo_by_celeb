"""See what every pipeline module does to a photo.

Runs the 12 modules on one camera photo + one celeb frame and writes an HTML
report: a before/after slider for each module (full photo + zoomed detail),
its measurements and warnings, and the final print/share files.

Usage:
    python preview_effects.py camera.jpg celeb.png
    python preview_effects.py camera.jpg celeb.png --config booth_config.json --out report
    python preview_effects.py camera.jpg celeb.png --ablation          # also: final photo WITHOUT each module
    python preview_effects.py camera.jpg celeb.png --disable edges,texture
    python preview_effects.py camera.jpg celeb.png --set color_match.strength=0.5 --set composite.space=gamma
    python preview_effects.py camera.jpg celeb.png --zoom 0.55,0.1,0.3,0.3   # detail area [x,y,w,h] 0..1

Then open <out>/index.html in a browser.
"""
import argparse
import html
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from booth.config import load_config
from booth.imaging import to_pil
from booth.modules import BY_KEY, MODULES
from booth.modules.m11_export import save_outputs
from booth.pipeline import run

PREVIEW_MAX = 1400
CANVAS_KEYS = {m.KEY for m in MODULES} - {"export", "quality"}


def parse_set(items):
    """--set a.b=value  ->  {"a": {"b": value}} (value parsed as JSON when possible)."""
    out = {}
    for item in items:
        path, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        *parents, last = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return out


def detail_box(ctx, size, zoom):
    w, h = size
    if zoom:
        x, y, zw, zh = zoom
        return round(x * w), round(y * h), round((x + zw) * w), round((y + zh) * h)
    f = ctx.celeb_face
    if f:  # celeb head + shoulder edge, where halos and texture show most
        side = int(f["h"] * 2.6)
        x0, y0 = f["cx"] - side // 2, f["y"] - side // 4
    else:
        side = min(w, h) // 3
        x0, y0 = int(w * 0.6), int(h * 0.1)
    x0, y0 = max(0, min(x0, w - side)), max(0, min(y0, h - side))
    return x0, y0, min(w, x0 + side), min(h, y0 + side)


def save_preview(img, path):
    img = img.copy()
    img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.Resampling.LANCZOS)
    img.save(path, quality=90)
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)


def save_detail(img, box, path):
    crop = img.crop(box)
    if crop.width < 500:
        f = 500 / crop.width
        crop = crop.resize((500, round(crop.height * f)), Image.Resampling.NEAREST if f >= 2 else Image.Resampling.LANCZOS)
    crop.save(path, quality=92)
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)


def contact_sheet(items, path, cols=4, cell=360):
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    rows = (len(items) + cols - 1) // cols
    thumbs = []
    for label, img in items:
        t = img.copy()
        t.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        thumbs.append((label, t))
    th = max(t.height for _, t in thumbs) + 34
    sheet = Image.new("RGB", (cols * (cell + 12) + 12, rows * (th + 12) + 12), (32, 32, 36))
    draw = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(thumbs):
        x, y = 12 + (i % cols) * (cell + 12), 12 + (i // cols) * (th + 12)
        sheet.paste(t, (x + (cell - t.width) // 2, y + 30))
        draw.text((x, y + 4), label, fill=(235, 235, 240), font=font)
    sheet.save(path, quality=90)


def calibration_image(img, regions, path, face=None):
    """Photo with a 0..1 grid and the configured regions, to read [x, y, w, h] values from."""
    img = img.convert("RGB").copy()
    img.thumbnail((900, 900), Image.Resampling.LANCZOS)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default(size=max(11, h // 70))
    except TypeError:
        font = ImageFont.load_default()
    for i in range(1, 10):
        x, y = round(w * i / 10), round(h * i / 10)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 90), width=1)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 90), width=1)
        draw.text((x + 2, 2), f".{i}", fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((2, y + 2), f".{i}", fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    for name, rect, color in regions:
        if rect:
            x, y, rw, rh = rect
            box = [x * w, y * h, (x + rw) * w, (y + rh) * h]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[3] + 3), name, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    if face:
        f = face["scale"]
        box = [face["x"] * f, face["y"] * f, (face["x"] + face["w"]) * f, (face["y"] + face["h"]) * f]
        draw.rectangle(box, outline=(255, 210, 0), width=2)
        draw.text((box[0], box[1] - h // 50), "face", fill=(255, 210, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path, quality=90)
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)


def on_checker(rgba):
    """RGBA image over a gray checkerboard, so transparent areas are visible."""
    w, h = rgba.size
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((xx // 24) + (yy // 24)) % 2 == 0, 150, 110).astype(np.uint8)
    bg = Image.fromarray(np.stack([board] * 3, -1), "RGB").convert("RGBA")
    return Image.alpha_composite(bg, rgba.convert("RGBA")).convert("RGB")


def fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def compare(before, after, label_b="before", label_a="after", note=""):
    return f"""<div class="cmp" data-split="50">
  <img src="{after}" alt="{label_a}"><img class="b" src="{before}" alt="{label_b}">
  <span class="tag l">{html.escape(label_b)}</span><span class="tag r">{html.escape(label_a)}</span>
  <input type="range" min="0" max="100" value="50" aria-label="before/after split">
</div>{f'<p class="note">{note}</p>' if note else ''}"""


def side_by_side(before, after, label_b, label_a):
    return f"""<div class="sbs"><figure><img src="{before}"><figcaption>{label_b}</figcaption></figure>
<figure><img src="{after}"><figcaption>{label_a}</figcaption></figure></div>"""


CSS = """
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--pass:#1f8a4c;--warn:#b7791f;--fail:#c53030;--accent:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141416;--card:#1d1d20;--fg:#ececf0;--muted:#9a9aa2;--line:#2c2c31;--accent:#8ea2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:0}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:18px 0}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.num{font-variant-numeric:tabular-nums;color:var(--muted)}
.pill{font-size:12px;font-weight:600;padding:2px 9px;border-radius:99px;border:1px solid currentColor}
.pass{color:var(--pass)}.warn{color:var(--warn)}.fail{color:var(--fail)}.skip{color:var(--muted)}
.doc{color:var(--muted);white-space:pre-line;margin:6px 0 12px;font-size:14px}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.cmp{position:relative;line-height:0;border-radius:8px;overflow:hidden;background:#000;user-select:none}
.cmp img{width:100%;height:auto;display:block}.cmp img.b{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}
.cmp input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}
.cmp::after{content:"";position:absolute;top:0;bottom:0;left:var(--x,50%);width:2px;background:#fff;box-shadow:0 0 4px #0008;pointer-events:none}
.tag{position:absolute;top:8px;font:600 12px/1 system-ui;color:#fff;background:#0009;padding:4px 7px;border-radius:5px;pointer-events:none}.tag.l{left:8px}.tag.r{right:8px}
.label{font-size:12px;color:var(--muted);margin:10px 0 4px;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top;word-break:break-word}
td:first-child{color:var(--muted);white-space:nowrap}ul.w{margin:8px 0 0;padding-left:18px;color:var(--warn);font-size:14px}
.sbs{display:grid;grid-template-columns:1fr 1fr;gap:10px}.sbs img{width:100%;border-radius:8px}figure{margin:0}figcaption{font-size:13px;color:var(--muted)}
.note{font-size:13px;color:var(--muted)}a{color:var(--accent)}nav{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
nav a{font-size:13px;text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:3px 8px;background:var(--card)}
"""

JS = """
document.querySelectorAll('.cmp').forEach(c=>{const r=c.querySelector('input'),b=c.querySelector('img.b');
const set=v=>{b.style.clipPath=`inset(0 ${100-v}% 0 0)`;c.style.setProperty('--x',v+'%')};r.addEventListener('input',e=>set(e.target.value));set(r.value)});
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("camera")
    p.add_argument("celeb")
    p.add_argument("--config", default="booth_config.json" if os.path.exists("booth_config.json") else None)
    p.add_argument("--celeb-name", help="settings key in config.celebs (default: celeb file name)")
    p.add_argument("--mask", help="optional user mask (white = user)")
    p.add_argument("--out", default="effects_report")
    p.add_argument("--disable", default="", help="comma separated module keys to switch off")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--ablation", action="store_true", help="also render the final photo without each module")
    p.add_argument("--zoom", help="detail area x,y,w,h (0..1 of the photo)")
    args = p.parse_args()

    disable = [k.strip() for k in args.disable.split(",") if k.strip()]
    unknown = [k for k in disable if k not in BY_KEY]
    if unknown:
        sys.exit(f"unknown module(s): {unknown}. Choose from: {', '.join(BY_KEY)}")
    for k in disable:
        if BY_KEY[k].REQUIRED:
            print(f"note: '{k}' is required and always runs (change its settings instead)")
    config = load_config(args.config, parse_set(args.set))
    zoom = [float(v) for v in args.zoom.split(",")] if args.zoom else None

    for sub in ("stages", "detail", "ablation"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    ctx, shots = run(args.camera, args.celeb, config, celeb_name=args.celeb_name, user_mask=args.mask,
                     disable=disable, snapshots=True)
    canvas_size = (ctx.camera.shape[1], ctx.camera.shape[0])
    box = detail_box(ctx, canvas_size, zoom)

    raw = Image.open(args.camera).convert("RGB")
    raw = raw.resize(canvas_size) if raw.size != canvas_size and not config["input"]["rotate"] % 180 else raw
    prev = {"image": raw, "canvas": raw.size == canvas_size}
    sheet = [("00 camera (raw)", raw)]
    cards, nav = [], []

    for i, shot in enumerate(shots, 1):
        key, img = shot["key"], shot["image"]
        is_canvas = key in CANVAS_KEYS or "print" not in ctx.outputs
        after = save_preview(img, f"{args.out}/stages/{i:02d}_{key}.jpg")
        before = save_preview(prev["image"], f"{args.out}/stages/{i:02d}_{key}_before.jpg")
        sheet.append((f"{i:02d} {shot['title']}", img))

        if shot["skipped"]:
            status, view = '<span class="pill skip">off</span>', '<p class="note">Module switched off: image unchanged.</p>'
        else:
            status = ""
            if is_canvas and prev["canvas"] and img.size == prev["image"].size:
                view = compare(before, after)
                if key not in ("input", "quality"):
                    d_b = save_detail(prev["image"], box, f"{args.out}/detail/{i:02d}_{key}_before.jpg")
                    d_a = save_detail(img, box, f"{args.out}/detail/{i:02d}_{key}.jpg")
                    view += '<div class="label">Detail (zoomed)</div>' + compare(d_b, d_a)
            elif key == "quality":
                view = f'<img src="{after}" alt="print" style="width:100%;border-radius:8px">'
            else:
                view = side_by_side(before, after, "before", "after")
        if key == "quality":
            st = shot["metrics"].get("status", "pass")
            status = f'<span class="pill {st}">{st}</span>'

        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(fmt(v))}</td></tr>"
                       for k, v in shot["metrics"].items())
        warns = "".join(f"<li>{html.escape(w)}</li>" for w in shot["warnings"])
        doc = html.escape((BY_KEY[key].__doc__ or "").strip())
        nav.append(f'<a href="#{key}">{i:02d} {html.escape(shot["title"].split(" (")[0])}</a>')
        cards.append(f"""<section class="card" id="{key}">
<div class="head"><span class="num">{i:02d}</span><h2>{html.escape(shot['title'])}</h2>{status}
<span class="note">{shot['ms']} ms · config: <code>{key}</code></span></div>
<p class="doc">{doc}</p>
<div class="grid"><div>{view}</div><div><div class="label">Measurements</div><table>{rows or '<tr><td>-</td><td></td></tr>'}</table>
{f'<ul class="w">{warns}</ul>' if warns else ''}</div></div></section>""")
        if not shot["skipped"]:
            prev = {"image": img, "canvas": is_canvas}

    os.makedirs(f"{args.out}/calibration", exist_ok=True)
    cam_img = shots[0]["image"]
    scale = min(900 / cam_img.width, 900 / cam_img.height, 1)
    face = dict(ctx.user_face, scale=scale) if ctx.user_face else None
    cal_cam = calibration_image(cam_img, [("reference_patch", config["scene"]["reference_patch"], (0, 230, 120)),
                                          ("wall", config["scene"]["wall"], (80, 160, 255))],
                                f"{args.out}/calibration/camera.jpg", face)
    cal_celeb = calibration_image(on_checker(to_pil(ctx.celeb)),
                                  [("neutral_region", ctx.celeb_info.get("neutral_region"), (0, 230, 120))],
                                  f"{args.out}/calibration/celeb.jpg")

    paths = save_outputs(ctx, config, args.out, "final")
    contact_sheet(sheet, f"{args.out}/contact_sheet.jpg")

    ablation = ""
    if args.ablation:
        full = shots[[s["key"] for s in shots].index("final_grade")]["image"]
        full_path = save_preview(full, f"{args.out}/ablation/all_on.jpg")
        d_full = save_detail(full, box, f"{args.out}/ablation/all_on_detail.jpg")
        parts = []
        for m in MODULES:
            if m.REQUIRED or m.KEY in ("export", "quality") or m.KEY in disable:
                continue
            print(f"ablation: without {m.KEY} ...")
            c = run(args.camera, args.celeb, config, celeb_name=args.celeb_name, user_mask=args.mask,
                    disable=disable + [m.KEY])
            img = to_pil(c.final if c.final is not None else c.composite)
            path = save_preview(img, f"{args.out}/ablation/without_{m.KEY}.jpg")
            d_without = save_detail(img, box, f"{args.out}/ablation/without_{m.KEY}_detail.jpg")
            parts.append(f"""<section class="card"><div class="head"><h2>Without: {html.escape(m.TITLE)}</h2></div>
<div class="grid"><div>{compare(path, full_path, 'without', 'all on')}</div><div>{compare(d_without, d_full, 'without', 'all on')}</div></div></section>""")
        ablation = '<h2 id="ablation" style="margin-top:36px">What each module adds (final photo with one module off)</h2>' + "".join(parts)
        nav.append('<a href="#ablation">Ablation</a>')
    nav.insert(0, '<a href="#calibration">Calibration</a>')

    final_idx = [m.KEY for m in MODULES].index("final_grade") + 1
    q = ctx.metrics.get("quality", {})
    st = q.get("status", "n/a")
    issues = "".join(f"<li>{html.escape(x)}</li>" for x in q.get("fails", []) + q.get("warnings", []))
    links = " · ".join(f'<a href="{os.path.basename(v)}">{k}</a>' for k, v in paths.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Booth Effects Report</title><style>{CSS}</style></head><body><main>
<h1>Booth effects report</h1>
<p class="sub">{html.escape(os.path.basename(args.camera))} + {html.escape(os.path.basename(args.celeb))} · config: {html.escape(str(args.config))}
{(' · disabled: ' + html.escape(', '.join(disable))) if disable else ''}</p>
<section class="card"><div class="head"><h2>Result</h2><span class="pill {st}">{st}</span></div>
{f'<ul class="w">{issues}</ul>' if issues else '<p class="note">No quality issues found.</p>'}
<p>Files: {links} · <a href="contact_sheet.jpg">contact sheet</a></p>
{compare('stages/01_input.jpg', f'stages/{final_idx:02d}_final_grade.jpg', 'camera', 'final')}
</section>
<section class="card" id="calibration"><div class="head"><h2>Calibration helper</h2></div>
<p class="doc">Read rectangles as [x, y, w, h] from the grid (0..1 of the image) and put them in the config:
left image = camera → scene.reference_patch (gray/white card) and scene.wall (plain wall, nobody walks there);
right image = celeb frame → celebs.&lt;name&gt;.neutral_region (something white or gray on your friend, e.g. the shirt).
Drawn boxes show what the current config uses; the yellow box is the detected face.
For exposure.target use the "measured" value of the Exposure module on an empty-booth photo.</p>
<div class="sbs"><figure><img src="{cal_cam}"><figcaption>camera (after input normalization)</figcaption></figure>
<figure><img src="{cal_celeb}"><figcaption>celeb frame</figcaption></figure></div></section>
<nav>{''.join(nav)}</nav>
{''.join(cards)}
{ablation}
</main><script>{JS}</script></body></html>"""
    with open(f"{args.out}/index.html", "w") as f:
        f.write(page)

    print(f"\nquality: {st.upper()}")
    for line in q.get("fails", []) + q.get("warnings", []):
        print("  -", line)
    for key, msg in ctx.warnings:
        print(f"  note [{key}]: {msg}")
    for shot in shots:
        flag = "off" if shot["skipped"] else f"{shot['ms']:>5} ms"
        print(f"  {shot['key']:<15} {flag}  {json.dumps(shot['metrics'])[:110]}")
    print(f"\nreport: {os.path.abspath(args.out)}/index.html")


if __name__ == "__main__":
    main()
