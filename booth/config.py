"""Default settings for every module. A JSON config file only needs the keys it changes."""
import copy
import json

DEFAULTS = {
    # Fixed things in the booth, as normalized [x, y, w, h] rectangles of the camera image.
    "scene": {
        "reference_patch": None,   # small white/gray card visible to the camera (outside the print crop)
        "wall": None,              # plain, evenly lit background area (no people walk there)
    },
    # Per-celeb settings, keyed by file name without extension; "default" is the fallback.
    "celebs": {
        "default": {
            "neutral_region": None,  # [x, y, w, h] on the celeb frame: something white/gray (shirt, card)
            "head_height": None,     # face box height in celeb frame pixels (None = detect)
            "eye_y": None,           # eye line in celeb frame pixels (None = detect)
            "height_ratio": 1.0,     # >1 when the celeb should look a bit bigger/closer than the user
            "cut_at_bottom": True,   # body is cut by the frame bottom: never lift it above the edge
        }
    },
    "input": {
        "rotate": 0,               # counter-clockwise degrees, like PIL (the old server used 90)
        "mirror": False,
        "working_height": 1800,    # resize the camera photo to this height (None = keep)
    },
    "white_balance": {
        "enabled": True,
        "method": "patch",         # patch | gray_pixels | calibration
        "gains": [1.0, 1.0, 1.0],  # used by "calibration" (linear RGB multipliers)
        "strength": 1.0,
        "max_gain": 2.0,
    },
    "exposure": {
        "enabled": True,
        "reference": "patch",      # patch | wall
        "target": None,            # brightness the reference should have, 0..1 sRGB (None = measure only)
        "max_gain": 2.0,
    },
    "scale_position": {
        "auto": True,              # match the user's head size and eye line (needs OpenCV)
        "base_height": 1.0,        # celeb frame height relative to the canvas when not auto
        "x": 0.72,                 # horizontal center of the celeb, 0 = left edge, 1 = right edge
        "max_change": 0.15,        # auto scale stays within +-15 % of the base size
        "eye_offset": 0.0,         # extra eye-line shift relative to the canvas height (negative = higher)
    },
    "color_match": {
        "enabled": True,
        "method": "neutral",       # neutral (match the light) | stats (old content-based match)
        "scene_region": "auto",    # patch | wall | auto
        "brightness": "manual",    # manual | neutral | none
        "brightness_gain": 1.0,    # used by "manual": calibrate once per event
        "strength": 0.9,
        "max_shift": 15.0,         # max color change (Lab a/b units) so a bad measurement can't wreck the photo
    },
    "light_shape": {
        "enabled": True,
        "auto": True,              # copy the left/right brightness balance of the user's face
        "direction_deg": 0,        # manual: where the light comes from, 0 = left, 90 = top, 180 = right
        "gradient": 0.0,           # manual: brightness difference across the celeb, 0.15 = +-15 %
        "max_gradient": 0.3,
        "contrast": 1.0,
    },
    "edges": {
        "enabled": True,
        "shrink_px": 1,
        "feather_px": 0.8,
        "decontaminate": True,     # replace edge colors with nearby inside colors (kills halos)
        "spill": "none",           # none | green | blue: remove screen color left in hair
        "light_wrap": 0.2,
        "light_wrap_px": 10,
    },
    "texture": {
        "enabled": True,
        "noise": "auto",           # auto | number (sigma, 0..1) | 0
        "noise_mono": 0.7,         # share of noise that is brightness-only (like real sensors)
        "blur": "auto",            # auto (compare faces) | number (sigma px) | 0
        "max_blur": 1.5,           # auto never blurs more than this (sigma px)
        "seed": 1,
    },
    "composite": {
        "space": "linear",         # linear (correct) | gamma (old way, dark edges)
        "user_in_front": True,     # needs a user mask
    },
    "final_grade": {
        "enabled": True,
        "lut": None,               # path to a .cube file
        "lut_strength": 1.0,
        "saturation": 1.05,
        "contrast": 1.05,
        "warmth": 0.0,             # + warmer, - cooler (Lab b units)
        "vignette": 0.12,
        "grain": 0.006,
        "seed": 2,
    },
    "export": {
        "enabled": True,
        "print_size_in": [4, 6],   # width, height in inches
        "dpi": 300,
        "crop_anchor": [0.5, 0.5],
        "print_brightness": 1.06,
        "print_saturation": 1.08,
        "sharpen": 0.6,
        "icc_profile": None,       # printer .icc/.icm file
        "frame": None,             # RGBA PNG drawn on top of the print
        "label": "AI-generated",
        "share_max_size": 1600,
        "jpeg_quality": 88,
    },
    "quality": {
        "enabled": True,
        "max_patch_cast": 4.0,
        "exposure_tolerance": 0.06,
        "max_color_delta": 6.0,
        "max_clipped": 0.02,
        "require_face": True,
        "check_eyes": True,
    },
}


def _merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    if overrides:
        _merge(cfg, overrides)
    return cfg


def celeb_settings(cfg, name):
    info = copy.deepcopy(DEFAULTS["celebs"]["default"])
    _merge(info, cfg["celebs"].get("default", {}))
    _merge(info, cfg["celebs"].get(name, {}))
    return info
