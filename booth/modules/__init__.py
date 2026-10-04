"""The 12 pipeline modules, in the order they run.

Every module has the same interface:
    KEY       config section name
    TITLE     human readable name
    REQUIRED  True = always runs (the photo can't be made without it)
    run(ctx, config) -> dict of measurements
"""
from . import (m01_input, m02_white_balance, m03_exposure, m04_scale_position, m05_color_match,
               m06_light_shape, m07_edges, m08_texture, m09_composite, m10_final_grade, m11_export,
               m12_quality)

MODULES = [m01_input, m02_white_balance, m03_exposure, m04_scale_position, m05_color_match,
           m06_light_shape, m07_edges, m08_texture, m09_composite, m10_final_grade, m11_export,
           m12_quality]

BY_KEY = {m.KEY: m for m in MODULES}
