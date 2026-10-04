"""Quality check: catch bad photos before they are printed.

Does not change the image. Reads the measurements of the other modules and
decides pass / warn / fail with reasons (moved camera, covered card, blink...).
"""
from ..imaging import count_open_eyes

KEY = "quality"
TITLE = "Quality check"
REQUIRED = False


def run(ctx, config):
    cfg, mx = config[KEY], ctx.metrics
    fails, warns = [], []

    wb = mx.get("white_balance", {})
    if wb.get("cast_after") is not None and wb["cast_after"] > cfg["max_patch_cast"]:
        warns.append(f"reference card still has a color cast ({wb['cast_after']}): card covered or moved?")

    ex = mx.get("exposure", {})
    target = config["exposure"]["target"]
    if target is not None and "after" in ex and abs(ex["after"] - target) > cfg["exposure_tolerance"]:
        warns.append(f"exposure off target ({ex['after']} vs {target}): gain limit reached?")
    if ex.get("clipped_fraction", 0) > cfg["max_clipped"]:
        warns.append(f"{ex['clipped_fraction']:.1%} of the camera photo is blown out: lower camera exposure")

    cm = mx.get("color_match", {})
    if cm.get("skipped"):
        warns.append("color match was skipped")
    elif cm.get("color_delta_after", 0) > cfg["max_color_delta"]:
        warns.append(f"celeb light color still differs from the room (delta {cm['color_delta_after']})")

    sp = mx.get("scale_position", {})
    if sp.get("clamped"):
        warns.append(f"user is too close/far (wanted scale x{sp['wanted_adjust']}): show 'step back/closer'")
    if sp.get("eye_line_limited"):
        warns.append("celeb could not be raised to the user's eye line (body is cut at the bottom)")

    eyes = None
    if not sp.get("face_detector", True):
        warns.append("face and blink checks skipped: OpenCV not installed")
    elif cfg["require_face"] and ctx.user_face is None:
        fails.append("no user face found: nobody in front of the camera, or face turned away")
    elif cfg["check_eyes"] and ctx.user_face:
        eyes = count_open_eyes(ctx.camera, ctx.user_face)
        if eyes is not None and eyes < 2:
            warns.append(f"only {eyes} open eye(s) found: blink? consider a retake")

    status = "fail" if fails else "warn" if warns else "pass"
    return {"status": status, "fails": fails, "warnings": warns, "open_eyes": eyes,
            "pipeline_warnings": [f"{k}: {msg}" for k, msg in ctx.warnings]}
