"""The object passed from module to module."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class Context:
    raw_camera: Image.Image                 # camera photo as received
    raw_celeb: Image.Image                  # celeb last frame (RGBA) as received
    celeb_info: dict                        # per-celeb settings from the config
    raw_user_mask: Optional[Image.Image] = None  # optional person mask from the frontend

    camera: Optional[np.ndarray] = None     # (H, W, 3) sRGB, set by 01_input
    celeb: Optional[np.ndarray] = None      # (h, w, 4) celeb source frame, set by 01_input
    user_mask: Optional[np.ndarray] = None  # (H, W) 0..1, user pixels
    celeb_layer: Optional[np.ndarray] = None  # (H, W, 4) celeb placed on the canvas, set by 04
    celeb_transform: Optional[dict] = None  # {"scale", "x", "y"}: canvas = source * scale + (x, y)
    user_face: Optional[dict] = None        # face box of the user on the canvas
    celeb_face: Optional[dict] = None       # face box of the celeb on the canvas
    composite: Optional[np.ndarray] = None  # (H, W, 3) set by 09_composite
    final: Optional[np.ndarray] = None      # (H, W, 3) set by 10_final_grade
    outputs: dict = field(default_factory=dict)   # name -> PIL image, set by 11_export
    metrics: dict = field(default_factory=dict)   # module key -> measurements
    warnings: list = field(default_factory=list)  # (module key, message)

    def warn(self, key, message):
        self.warnings.append((key, message))

    def celeb_rect_to_canvas(self, rect):
        """Map a normalized rect on the celeb source frame to a pixel box on the canvas."""
        h, w = self.celeb.shape[:2]
        t = self.celeb_transform
        x, y, rw, rh = rect
        return (int(round(y * h * t["scale"] + t["y"])), int(round((y + rh) * h * t["scale"] + t["y"])),
                int(round(x * w * t["scale"] + t["x"])), int(round((x + rw) * w * t["scale"] + t["x"])))
