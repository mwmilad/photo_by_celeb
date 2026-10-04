# Booth photo pipeline

Turns the **camera photo** (real user, full body) and the **celeb's last video frame** (RGBA)
into one believable photo, a **print** file and a **share** file.

```
camera photo ─► 01 input ─► 02 white balance ─► 03 exposure ─┐
                                                             ├─► 09 composite ─► 10 final grade ─► 11 export ─► 12 quality
celeb frame  ─► 04 scale & position ─► 05 color match ─► 06 light shape ─► 07 edges ─► 08 texture ─┘
```

The rule behind it: **the celeb adapts to the room, never the other way round.** The camera
photo is reality, so it is only corrected to be neutral (02, 03). Everything else changes the celeb.

## Install

```bash
pip install -r requirements.txt
```

OpenCV is optional. Without it the face-based features (auto scale, auto light shape,
auto blur, blink check) switch off and fall back to config values. Use OpenCV **< 5**;
version 5 no longer includes the face detection models.

## See what every module does

```bash
python preview_effects.py camera.jpg assets/friend.png
# then open effects_report/index.html
```

The report shows:
- a **before/after slider** for every module, on the full photo and a zoomed detail
  (edges and texture are only visible zoomed in)
- the **measurements** of each module (gains, color difference before/after, scale...)
- the **quality result** (pass / warn / fail) with reasons
- a **calibration helper**: a 0..1 grid over the camera photo and the celeb frame to read region values from
- `final_print.png` (with DPI), `final_share.jpg` and `contact_sheet.jpg`

| Option | What it does |
|---|---|
| `--config file.json` | settings (default: `booth_config.json`) |
| `--ablation` | also renders the final photo **without** each module, to see what each one adds |
| `--disable edges,texture` | switch modules off |
| `--set color_match.strength=0.5` | try a setting without editing the config (repeatable) |
| `--zoom 0.55,0.1,0.3,0.3` | detail area `x,y,w,h` (default: around the celeb's head) |
| `--mask user_mask.png` | person mask from the frontend: user is drawn in front of the celeb |
| `--celeb-name name` | which `celebs` entry to use (default: celeb file name) |

## The modules

Each module is one file in `booth/modules/` with the same interface: `run(ctx, config)`
changes the images in `ctx` and returns its measurements. Its settings live in the config
section with the same name. Defaults and comments for **every** setting are in `booth/config.py`.

| # | Module | Fixes | Main settings |
|---|---|---|---|
| 01 | `input` | rotation, mirroring, size, color profiles | `input.rotate`, `input.mirror`, `input.working_height` |
| 02 | `white_balance` | color cast of the room light | `scene.reference_patch`, `white_balance.method` |
| 03 | `exposure` | room brightness changing between photos | `exposure.target` |
| 04 | `scale_position` | celeb size and eye line vs. the user (#8) | `scale_position.*`, `celebs.<name>.height_ratio` |
| 05 | `color_match` | celeb light color ≠ room light (#5) | `celebs.<name>.neutral_region`, `color_match.*` |
| 06 | `light_shape` | light direction and contrast (#5) | `light_shape.*` |
| 07 | `edges` | halos, green/blue spill, cut-out look | `edges.*` |
| 08 | `texture` | celeb too sharp / too clean | `scene.wall`, `texture.*` |
| 09 | `composite` | dark edges from gamma blending; user in front | `composite.*` |
| 10 | `final_grade` | one shared look over both people (LUT, saturation, vignette, grain) | `final_grade.*` |
| 11 | `export` | print size, print boost, ICC profile, frame, AI label, share JPEG | `export.*` |
| 12 | `quality` | catches covered card, blown highlights, wrong distance, blinks | `quality.*` |

Modules 01, 04 and 09 always run (no photo without them); all others can be switched off
with `"enabled": false` or `--disable`.

## Calibrate once per event (about 10 minutes)

1. **Lock the camera**: fixed exposure and white balance.
2. Put a **gray or white card** where the camera sees it but **outside the print crop**.
3. Take a photo of the **empty booth** and run `preview_effects.py` on it.
   In the *Calibration helper*, read the card's rectangle from the grid → `scene.reference_patch`,
   and a plain wall area where nobody walks → `scene.wall`.
4. Copy the *measured* value of the Exposure module → `exposure.target`.
5. For each friend's video, run the report on its last frame and read a white/gray area of
   their clothes from the right-hand grid → `celebs.<file name>.neutral_region`.
   Without it the module searches for grayish pixels by itself (skipping skin), which works
   but is less reliable.
6. Take a photo **with a person** and check the report. Tune strengths with `--set`.
7. **Test print**, then adjust `export.print_brightness` / `export.print_saturation`
   or set `export.icc_profile` to your printer's profile.

Example `booth_config.json` after calibration:

```json
{
  "scene": { "reference_patch": [0.045, 0.045, 0.06, 0.03], "wall": [0.6, 0.05, 0.3, 0.2] },
  "input": { "rotate": 90, "mirror": true },
  "exposure": { "target": 0.62 },
  "edges": { "spill": "green" },
  "celebs": {
    "ali":   { "neutral_region": [0.42, 0.40, 0.15, 0.25] },
    "sara":  { "neutral_region": [0.38, 0.45, 0.20, 0.20], "height_ratio": 0.95 }
  }
}
```

`input.rotate: 90` and `input.mirror: true` copy what the old server did to the camera
photo. Change them if your camera delivers upright photos.

## Use it in the server

```python
from booth import load_config, run

CONFIG = load_config("booth_config.json")   # once, at startup

ctx = run(file.file, f"assets/{selected}.png", CONFIG)   # celeb settings = celebs["<selected>"]
if ctx.metrics["quality"]["status"] == "fail":
    ...  # ask for a retake
ctx.outputs["print"].save(print_path, dpi=(300, 300))
ctx.outputs["share"].save(share_path, quality=88)
```

Check `selected` against the list of files in `assets/` before using it in a path.

## Tests

```bash
python -m pytest tests
```
