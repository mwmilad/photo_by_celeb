import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from booth.config import load_config
from booth.context import Context
from booth.imaging import color_cast, linear_to_srgb, region_slices, srgb_to_linear, to_pil
from booth.modules import m02_white_balance, m03_exposure, m05_color_match, m07_edges, m09_composite
from booth.modules.m10_final_grade import apply_lut, load_cube
from booth.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
H, W = 600, 400
PATCH = [20 / W, 20 / H, 40 / W, 20 / H]  # gray card at x 20..60, y 20..40
WARM = (1.25, 1.0, 0.7)


def camera(cast=WARM):
    lin = np.full((H, W, 3), 0.3, np.float32)
    lin[20:40, 20:60] = 0.35
    return linear_to_srgb(lin * np.array(cast, np.float32))


def ctx_with(cam, layer=None, info=None):
    ctx = Context(raw_camera=None, raw_celeb=None, celeb_info=info or {})
    ctx.camera = cam
    if layer is not None:
        ctx.celeb = ctx.celeb_layer = layer
        ctx.celeb_transform = {"scale": 1.0, "x": 0, "y": 0}
    return ctx


def celeb_layer(cast=(0.82, 1.0, 1.22)):
    """Blue-tinted celeb: white shirt (rows 300..400) and skin (rows 150..250), opaque body."""
    lin = np.zeros((H, W, 3), np.float32)
    lin[100:, 100:300] = 0.05                 # dark jacket
    lin[300:400, 150:250] = 0.7               # white shirt
    lin[150:250, 150:250] = [0.55, 0.32, 0.25]  # skin
    rgb = linear_to_srgb(lin * np.array(cast, np.float32))
    alpha = np.zeros((H, W, 1), np.float32)
    alpha[100:, 100:300] = 1
    return np.concatenate([rgb, alpha], -1)


def patch_cast(img):
    px = img[region_slices(img.shape, PATCH)].reshape(-1, 3).mean(0)
    return color_cast(srgb_to_linear(px))


def test_white_balance_patch_removes_cast():
    ctx = ctx_with(camera())
    m = m02_white_balance.run(ctx, load_config(overrides={"scene": {"reference_patch": PATCH}}))
    assert m["cast_before"] > 10
    assert patch_cast(ctx.camera) < 1.0


def test_white_balance_without_patch_falls_back_and_warns():
    ctx = ctx_with(camera())
    m = m02_white_balance.run(ctx, load_config())
    assert m["method"] in ("gray_pixels", "gray_world")
    assert ctx.warnings and patch_cast(ctx.camera) < 2.0


def test_exposure_reaches_target_and_measures_without_target():
    cfg = load_config(overrides={"scene": {"reference_patch": PATCH}, "exposure": {"target": 0.6}})
    ctx = ctx_with(camera((1, 1, 1)))
    assert abs(m03_exposure.run(ctx, cfg)["after"] - 0.6) < 0.01

    ctx = ctx_with(camera((1, 1, 1)))
    before = ctx.camera.copy()
    m = m03_exposure.run(ctx, load_config(overrides={"scene": {"reference_patch": PATCH}}))
    assert "measured" in m and np.array_equal(before, ctx.camera)


def _shirt(layer):
    return srgb_to_linear(layer[300:400, 150:250, :3].reshape(-1, 3).mean(0))


def test_color_match_uses_neutral_region():
    cfg = load_config(overrides={"scene": {"reference_patch": PATCH}, "color_match": {"strength": 1.0}})
    cam = camera((1, 1, 1))
    ctx = ctx_with(cam, celeb_layer(), {"neutral_region": [150 / W, 300 / H, 100 / W, 100 / H]})
    m = m05_color_match.run(ctx, cfg)
    assert m["celeb_source"] == "config_region"
    assert m["color_delta_after"] < 1.0 < m["color_delta_before"]
    assert color_cast(_shirt(ctx.celeb_layer)) < 1.5


def test_color_match_auto_ignores_skin():
    cfg = load_config(overrides={"scene": {"reference_patch": PATCH}, "color_match": {"strength": 1.0}})
    ctx = ctx_with(camera((1, 1, 1)), celeb_layer())
    ctx.celeb_face = {"x": 150, "y": 150, "w": 100, "h": 100, "cx": 200, "eye_y": 190}
    before_skin = srgb_to_linear(ctx.celeb_layer[200, 200, :3])
    m = m05_color_match.run(ctx, cfg)
    assert m["celeb_source"] == "auto_gray_pixels"
    assert color_cast(_shirt(ctx.celeb_layer)) < 2.0           # the shirt became neutral
    after_skin = srgb_to_linear(ctx.celeb_layer[200, 200, :3])
    ratio = after_skin / before_skin                           # skin got the same light change...
    assert np.allclose(ratio, m["gains"], rtol=0.03)           # ...not a separate "skin match"


def test_color_match_respects_max_shift():
    cfg = load_config(overrides={"scene": {"reference_patch": PATCH},
                                 "color_match": {"strength": 1.0, "max_shift": 2.0}})
    ctx = ctx_with(camera((1, 1, 1)), celeb_layer(cast=(0.5, 1.0, 1.8)),
                   {"neutral_region": [150 / W, 300 / H, 100 / W, 100 / H]})
    m = m05_color_match.run(ctx, cfg)
    assert m["limited"] and color_cast(np.full(3, 0.18) * m["gains"]) <= 2.01


def test_linear_composite_has_no_dark_edges():
    cam = np.zeros((10, 10, 3), np.float32)
    layer = np.ones((10, 10, 4), np.float32)
    layer[..., 3] = 0.5  # half transparent white edge over black
    results = {}
    for space in ("linear", "gamma"):
        ctx = ctx_with(cam, layer)
        m09_composite.run(ctx, load_config(overrides={"composite": {"space": space}}))
        results[space] = float(ctx.composite.mean())
    assert abs(results["gamma"] - 0.5) < 1e-3
    assert results["linear"] > 0.7  # 50 % of the light = sRGB 0.735


def test_user_mask_puts_user_in_front():
    cam = np.zeros((10, 10, 3), np.float32)
    ctx = ctx_with(cam, np.ones((10, 10, 4), np.float32))
    ctx.user_mask = np.zeros((10, 10), np.float32)
    ctx.user_mask[:, :5] = 1
    m09_composite.run(ctx, load_config())
    assert ctx.composite[:, :5].max() == 0 and ctx.composite[:, 5:].min() > 0.99


def test_edges_remove_green_spill():
    layer = np.zeros((60, 60, 4), np.float32)
    layer[10:50, 10:50] = [0.5, 0.5, 0.5, 1.0]
    ring = np.zeros((60, 60), bool)
    ring[8:52, 8:52] = True
    ring[11:49, 11:49] = False
    layer[ring] = [0.2, 0.8, 0.2, 0.5]  # green fringe
    ctx = ctx_with(np.full((60, 60, 3), 0.5, np.float32), layer)
    m07_edges.run(ctx, load_config(overrides={"edges": {"spill": "green", "light_wrap": 0}}))
    out = ctx.celeb_layer
    edge = (out[..., 3] > 0) & (out[..., 3] < 0.98)
    assert edge.any()
    assert (out[edge, 1] <= np.maximum(out[edge, 0], out[edge, 2]) + 1e-4).all()


def test_identity_cube_lut(tmp_path):
    n = 5
    g = np.linspace(0, 1, n)
    rows = [f"{r} {gg} {b}" for b in g for gg in g for r in g]  # red changes fastest
    path = tmp_path / "id.cube"
    path.write_text(f"LUT_3D_SIZE {n}\n" + "\n".join(rows))
    img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert np.allclose(apply_lut(img, *load_cube(path)), img, atol=1e-5)


def write_inputs(tmp_path):
    cam = to_pil(camera())
    cam_path, celeb_path = tmp_path / "camera.png", tmp_path / "friend.png"
    cam.save(cam_path)
    rgba = celeb_layer()
    rgba[:100] = 0  # transparent head room
    to_pil(rgba).save(celeb_path)
    return cam_path, celeb_path


def test_pipeline_end_to_end(tmp_path):
    cam_path, celeb_path = write_inputs(tmp_path)
    cfg = load_config(overrides={"scene": {"reference_patch": PATCH}, "input": {"working_height": 900}})
    ctx = run(cam_path, celeb_path, cfg, disable=["edges", "composite"])
    assert "edges" not in ctx.metrics            # optional module switched off
    assert "composite" in ctx.metrics            # required module always runs
    assert ctx.outputs["print"].size == (1200, 1800)  # 4x6 inch at 300 dpi
    assert max(ctx.outputs["share"].size) <= 1600
    assert ctx.metrics["quality"]["status"] in ("pass", "warn", "fail")


def test_preview_script(tmp_path):
    cam_path, celeb_path = write_inputs(tmp_path)
    out = tmp_path / "report"
    res = subprocess.run([sys.executable, str(ROOT / "preview_effects.py"), str(cam_path), str(celeb_path),
                          "--out", str(out), "--disable", "texture", "--set", "composite.space=gamma",
                          "--set", "input.working_height=600"],
                         capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    page = (out / "index.html").read_text()
    assert "White balance" in page and 'id="texture"' in page and "off</span>" in page
    assert (out / "contact_sheet.jpg").exists() and (out / "final_print.png").exists()
    assert Image.open(out / "final_print.png").info.get("dpi", (0,))[0] > 299
