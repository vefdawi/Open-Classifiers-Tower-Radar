"""
Takes LEVEL INDEX (0..13) of each of the 8 transformation dimensions 
and writes the resulting image to a file.

251 px wide × 795 px tall (INPUT_W × INPUT_H), 8-bit grayscale PNG

"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

INPUT_H = 795          # canvas height (px)
INPUT_W = 251          # canvas width  (px)
N_STEPS = 14           # valid level index 0..13

RX_REF = 30            # semi-axis width
RY_REF = 120           # semi-axis length scaled by 'scale'

OCC_RX_RATIO = 0.85    # occluder is slightly smaller than the focal ellipse
OCC_RY_RATIO = 0.85

# 14-point grids
GRIDS = {
    "bg_contrast":  np.linspace(1.0, 0.1, N_STEPS),                  # 1.0 to 0.1
    "bg_block":     np.arange(1, N_STEPS + 1).astype(float),         # 1 to 14
    "occ_offset_x": np.linspace(55.0, 9.5, N_STEPS),                 # 55.0 to 9.5
    "translate_x":  np.linspace(126.0, 165.0, N_STEPS) - (INPUT_W // 2),
    "translate_y":  np.linspace(137.0, 397.0, N_STEPS) - (INPUT_H // 2),
    "scale":        np.linspace(0.4, 1.0, N_STEPS),                  # 0.4 to 1.0
    "angle_deg":    np.linspace(0.0, 45.5, N_STEPS),                 # 0 to 45.5°
    "blur_sigma":   np.linspace(0.0, 13.0, N_STEPS),                 # 0 to 13
}

LEVEL_NAMES = list(GRIDS.keys())

NEUTRAL_LEVELS = {
    "bg_contrast":  0,    # 1.0
    "bg_block":     0,    # 1
    "occ_offset_x": 0,    # 55.0  (occluder far right → no visible occlusion)
    "translate_x":  0,    # +1.0 px  (≈ centred)
    "translate_y":  13,   # 0.0     (centred)
    "scale":        13,   # 1.0     (full length)
    "angle_deg":    0,    # 0°
    "blur_sigma":   0,    # 0
}


def _value(name: str, level: int) -> float:
    """Map a level index (0..13) to the parameter value on its grid."""
    lv = int(level)
    if lv != level or not (0 <= lv < N_STEPS):
        raise ValueError(
            f"{name}: level must be an integer in 0..{N_STEPS - 1}, got {level!r}"
        )
    return float(GRIDS[name][lv])


def _make_background(bg_contrast: float, bg_block: float,
                     bg_seed: int | None = None) -> np.ndarray:
    """Fine random noise, block-coarsened, contrast faded around the mean."""
    fine = np.random.default_rng(bg_seed).random(
        (INPUT_H, INPUT_W)).astype(np.float32)

    block = max(1, int(round(bg_block)))
    if block == 1:
        coarse = fine.copy()
    else:
        coarse = np.zeros_like(fine)
        for y in range(0, INPUT_H, block):
            for x in range(0, INPUT_W, block):
                coarse[y:y + block, x:x + block] = float(
                    fine[y:y + block, x:x + block].mean())

    mean = float(coarse.mean())
    return (mean + bg_contrast * (coarse - mean)).clip(0, 1)


def _ellipse_mask(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Boolean mask for an ellipse."""
    ys, xs = np.mgrid[0:INPUT_H, 0:INPUT_W]
    return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0

def render_scene(
    bg_contrast:  int = NEUTRAL_LEVELS["bg_contrast"],
    bg_block:     int = NEUTRAL_LEVELS["bg_block"],
    occ_offset_x: int = NEUTRAL_LEVELS["occ_offset_x"],
    translate_x:  int = NEUTRAL_LEVELS["translate_x"],
    translate_y:  int = NEUTRAL_LEVELS["translate_y"],
    scale:        int = NEUTRAL_LEVELS["scale"],
    angle_deg:    int = NEUTRAL_LEVELS["angle_deg"],
    blur_sigma:   int = NEUTRAL_LEVELS["blur_sigma"],
    out_path:     str = "scene.png",
    bg_seed:      int | None = None,
) -> str:
    """
    Render one crescent-ellipse scene and save it as a PNG.

    Every one of the 8 dimensions is given as a LEVEL INDEX in 0..13, which is
    looked up in the corresponding 14-point grid in `GRIDS`.

    Pipeline (strictly ordered):
      1. background : contrast fading + block coarsening
      2. crescent   : focal ellipse with occluder in a shared reference frame
      3. translation: shifts focal + occluder together
      4. scale      : scales RY of both (RX fixed), centres unchanged
      5. rotation   : rotates the crescent as a rigid body around the focal centre
      6. edge blur  : gaussian blur of the crescent, then composited on background

    Parameters
    ----------
    out_path : where to write the image (extension decides the format; PNG
               recommended). Written at exactly INPUT_W by INPUT_H px, 8-bit
               grayscale, no axes or margins.
    bg_seed  : seed for the background noise. None (default): new noise on every call.

    Returns
    -------
    str : `out_path`.
    """
    # levels to values
    v_bg_contrast  = _value("bg_contrast",  bg_contrast)
    v_bg_block     = _value("bg_block",     bg_block)
    v_occ_offset_x = _value("occ_offset_x", occ_offset_x)
    v_translate_x  = _value("translate_x",  translate_x)
    v_translate_y  = _value("translate_y",  translate_y)
    v_scale        = _value("scale",        scale)
    v_angle_deg    = _value("angle_deg",    angle_deg)
    v_blur_sigma   = _value("blur_sigma",   blur_sigma)

    # 1 background
    background = _make_background(v_bg_contrast, v_bg_block, bg_seed)

    # 2 to 4 geometry: centre, scaled semi-axes, occluder placement
    cx = INPUT_W // 2 + v_translate_x
    cy = INPUT_H // 2 + v_translate_y

    ry_focal = RY_REF * v_scale
    rx_focal = RX_REF                      # width always fixed
    ry_occ   = ry_focal * OCC_RY_RATIO
    rx_occ   = rx_focal * OCC_RX_RATIO

    focal_mask = _ellipse_mask(cx, cy, rx_focal, ry_focal)
    occ_mask   = _ellipse_mask(cx + v_occ_offset_x, cy, rx_occ, ry_occ)
    crescent   = (focal_mask & ~occ_mask).astype(np.float32)

    # 5 rotation of the crescent around the focal centre
    if v_angle_deg != 0.0:
        ys, xs = np.mgrid[0:INPUT_H, 0:INPUT_W]
        dx, dy = xs - cx, ys - cy
        a = np.deg2rad(v_angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        src_xi = np.round((dx * ca + dy * sa) + cx).astype(int)
        src_yi = np.round((-dx * sa + dy * ca) + cy).astype(int)
        valid = ((src_xi >= 0) & (src_xi < INPUT_W)
                 & (src_yi >= 0) & (src_yi < INPUT_H))
        rotated = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
        rotated[valid] = crescent[src_yi[valid], src_xi[valid]]
        crescent = rotated

    # 6 edge blur
    if v_blur_sigma > 0:
        soft = gaussian_filter(crescent, sigma=v_blur_sigma)
        soft = soft / soft.max() if soft.max() > 0 else soft
    else:
        soft = crescent

    # composite crescent (white) over background
    canvas = (soft + (1.0 - soft) * background).clip(0, 1)

    # write the raw canvas
    Image.fromarray((canvas * 255).astype(np.uint8), mode="L").save(out_path)
    return out_path


if __name__ == "__main__":
    render_scene(out_path="scene_neutral.png")
    render_scene(bg_contrast=10, bg_block=1, occ_offset_x=11, translate_x=12,
                 translate_y=13, scale=11, angle_deg=3, blur_sigma=2,
                 out_path="scene_mixed.png")
    print("wrote scene_neutral.png and scene_mixed.png")
