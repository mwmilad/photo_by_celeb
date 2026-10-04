"""Photo booth compositing pipeline: camera photo + celeb frame -> print and share images.

    from booth import load_config, run
    ctx = run("camera.jpg", "assets/friend.png", load_config("booth_config.json"))
    ctx.outputs["print"].save("print.png")
"""
from .config import load_config
from .pipeline import run

__all__ = ["load_config", "run"]
