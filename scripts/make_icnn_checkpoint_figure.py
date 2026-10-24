"""Rebuild the ICNN checkpoint figure (TCI revision) from the composited PNGs.

The raw reconstructions were not stored; the composited rows contain each
checkpoint at native resolution (192 px wide), so zoomed crops can be cut out
without resampling artefacts beyond the chosen nearest-neighbour magnification.
Outputs individual tiles that are laid out in LaTeX (so labels use the paper font).
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

SRC = sys.argv[1] if len(sys.argv) > 1 else "Figs/Checkpoints/ICNN"
OUT = os.path.join(SRC, "tiles")
os.makedirs(OUT, exist_ok=True)

ROWS = {
    "isgd_q0":   "row_val_vis_logs_0.01_1.0_1_0_ICNN.png",
    "isgd_q05":  "row_val_vis_logs_0.01_1.0_1_0.5_ICNN.png",
    "iadam_q0":  "row_val_vis_logs_0.01_0.01_0.25_0_ICNN_adam.png",
    "iadam_q05": "row_val_vis_logs_0.01_1.0_1_0.5_ICNN_adam.png",
}
# PSNR values printed on the original tiles (read off the composited images)
PSNR = {
    "isgd_q0":   [23.95, 24.02, 26.55, 27.13, 27.05],
    "isgd_q05":  [25.46, 26.60, 27.31, 27.51, 27.75],
    "iadam_q0":  [27.45, 27.58, 28.04, 28.02, 28.20],
    "iadam_q05": [26.34, 26.93, 27.46, 27.79, 28.13],
}
GTNOISY = "val_gt_noisy_col_vis_logs_0.01_1.0_1_0.5_ICNN.png"
NOISY_PSNR = 20.17

TILE_X0 = [10 + 195 * c for c in range(5)]   # left edge of each checkpoint tile
MAIN_Y0, MAIN_H, MAIN_W = 78, 190, 192        # main image inside each tile
ZOOM = (138, 58, 48)                          # (x, y, size) in main-image pixels
MAG = 8                                       # nearest-neighbour magnification of saved crops


def main_image(row_png, c):
    im = Image.open(os.path.join(SRC, row_png)).convert("RGB")
    x0 = TILE_X0[c]
    return np.asarray(im.crop((x0, MAIN_Y0, x0 + MAIN_W, MAIN_Y0 + MAIN_H))).astype(float)


# Registration of the GT/noisy panel (rendered at 188 px) to the checkpoint tiles: a grid
# search over scale (184-196 px) and integer offset (+-8 px), maximising the correlation of
# image gradients on the zoom region against three different reconstructions, consistently
# selects a 192-px resampling with offset (dx, dy) = (2, 1).
GT_SIZE, GT_DX, GT_DY = 192, 2, 1


def to_tile_grid(img_small):
    return np.asarray(Image.fromarray(img_small.astype(np.uint8)).resize((GT_SIZE, GT_SIZE), Image.BICUBIC)).astype(float)


def font(px):
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans.ttf")
    return ImageFont.truetype(path, px)


def label(img, text, frac=0.15):
    """Burn a PSNR label into the top-left corner (white on translucent black)."""
    img = img.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = font(int(img.size[1] * frac))
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    pad = int(img.size[1] * 0.02)
    d.rectangle([0, 0, r - l + 2 * pad, b - t + 2 * pad], fill=(0, 0, 0, 170))
    d.text((pad - l, pad - t), text, font=f, fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, over).convert("RGB")


def crop(arr, x, y, s):
    return Image.fromarray(arr[y:y + s, x:x + s].astype(np.uint8)).resize((s * MAG, s * MAG), Image.NEAREST)


if __name__ == "__main__":
    x, y, s = ZOOM
    # checkpoint crops
    for key, png in ROWS.items():
        for c in range(5):
            m = main_image(png, c)
            sub = m[y:y + s, x:x + s]
            red = (sub[..., 0] > 190) & (sub[..., 1] < 80) & (sub[..., 2] < 80)
            assert red.sum() == 0, f"zoom region hits the red overlay in {png} tile {c}"
            label(crop(m, x, y, s), f"{PSNR[key][c]:.2f}").save(os.path.join(OUT, f"{key}_c{c}.png"))

    # ground truth and noisy input, registered to the reconstruction tiles
    col = np.asarray(Image.open(os.path.join(SRC, GTNOISY)).convert("RGB")).astype(float)
    gt_small, noisy_small = col[9:197, 10:198], col[261:449, 10:198]
    gt_big, noisy_big = to_tile_grid(gt_small), to_tile_grid(noisy_small)

    for name, a in [("gt", gt_big), ("noisy", noisy_big)]:
        gx, gy = x + GT_DX, y + GT_DY
        crop(a, gx, gy, s).save(os.path.join(OUT, f"{name}_zoom.png") if name == "gt" else os.path.join(OUT, "noisy_zoom_raw.png"))
        full = Image.fromarray(a.astype(np.uint8)).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST)
        d = ImageDraw.Draw(full)
        d.rectangle([gx * 4, gy * 4, (gx + s) * 4 - 1, (gy + s) * 4 - 1], outline=(255, 0, 0), width=8)
        full.save(os.path.join(OUT, f"{name}_full.png"))
    label(Image.open(os.path.join(OUT, "noisy_zoom_raw.png")), f"{NOISY_PSNR:.2f}").save(os.path.join(OUT, "noisy_zoom.png"))
    os.remove(os.path.join(OUT, "noisy_zoom_raw.png"))
    print("tiles written to", OUT)
