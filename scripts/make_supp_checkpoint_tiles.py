"""Tiles for the supplementary checkpoint figures (TCI revision), same style as Fig. 1.

Denoising: zoomed crops of the lettering on the boat (inside the red box of the original
composited tiles, so no overlay pixels), cut at native resolution and enlarged with
nearest-neighbour; GT/noisy resampled to the tile grid (190 px) as for the ICNN figure.
Inpainting: the composited GT/noisy panel is split into its four sub-images.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from make_icnn_checkpoint_figure import label  # same PSNR label style

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Figs/Checkpoints"

# ---------------- denoising (CRR) ----------------
D = os.path.join(ROOT, "Denoise")
OUT = os.path.join(D, "tiles"); os.makedirs(OUT, exist_ok=True)
ROWS = {"eps1e-2_p05": "row_val_vis_logs_0.01_0.01_0.5_0.png",
        "eps1_p2": "row_val_vis_logs_0.01_1.0_2_0.png"}
PSNR = {"eps1e-2_p05": [25.74, 27.91, 28.17, 28.54],
        "eps1_p2": [24.89, 28.33, 28.66, 28.94]}
NOISY_PSNR = 20.22                    # printed on the original GT/noisy panel
Y0, H, W = 79, 190, 192               # main image inside each composited tile
ZX, ZY, ZW, ZH = 83, 162, 74, 25      # interior of the red box (lettering), main-image px
MAG = 6

def crop(a):
    return Image.fromarray(a[ZY:ZY + ZH, ZX:ZX + ZW].astype(np.uint8)).resize((ZW * MAG, ZH * MAG), Image.NEAREST)

for key, png in ROWS.items():
    im = np.asarray(Image.open(os.path.join(D, png)).convert("RGB")).astype(float)
    for c in range(4):
        x0 = 10 + 195 * c
        m = im[Y0:Y0 + H, x0:x0 + W]
        sub = m[ZY:ZY + ZH, ZX:ZX + ZW]
        assert (((sub[..., 0] > 190) & (sub[..., 1] < 80) & (sub[..., 2] < 80)).sum() == 0), (png, c)
        label(crop(m), f"{PSNR[key][c]:.2f}", frac=0.3).save(os.path.join(OUT, f"{key}_c{c}.png"))

# GT/noisy registration (same procedure as for the ICNN figure): 192-px resampling with
# offset (dx, dy) = (1, 2), consistent across both configurations.
GT_SIZE, GT_DX, GT_DY = 192, 1, 2
col = np.asarray(Image.open(os.path.join(D, "val_gt_noisy.png")).convert("RGB")).astype(np.uint8)
for name, sl in [("gt", col[9:197, 10:198]), ("noisy", col[261:449, 10:198])]:
    a = np.asarray(Image.fromarray(sl).resize((GT_SIZE, GT_SIZE), Image.BICUBIC)).astype(float)
    a_sh = np.roll(np.roll(a, -GT_DY, 0), -GT_DX, 1)      # align to the tile grid
    z = crop(a_sh)
    (label(z, f"{NOISY_PSNR:.2f}", frac=0.3) if name == "noisy" else z).save(os.path.join(OUT, f"{name}_zoom.png"))
    full = Image.fromarray(a.astype(np.uint8)).resize((GT_SIZE * 4, GT_SIZE * 4), Image.NEAREST)
    bx, by = ZX + GT_DX, ZY + GT_DY
    ImageDraw.Draw(full).rectangle([bx * 4, by * 4, (bx + ZW) * 4 - 1, (by + ZH) * 4 - 1], outline=(255, 0, 0), width=8)
    full.save(os.path.join(OUT, f"{name}_full.png"))

# ---------------- inpainting (CRR) ----------------
P = os.path.join(ROOT, "inpainting")
OUT = os.path.join(P, "tiles"); os.makedirs(OUT, exist_ok=True)
g = Image.open(os.path.join(P, "gt_noisy.png")).convert("RGB")
for name, box in [("gt_full", (30, 30, 509, 508)), ("gt_zoom", (659, 30, 1492, 508)),
                  ("noisy_full", (30, 590, 509, 1068)), ("noisy_zoom", (659, 590, 1492, 1068))]:
    t = g.crop(box)
    a = np.asarray(t).astype(int)
    nz = np.nonzero((a.sum(-1) < 740))           # trim residual white margins
    t = t.crop((nz[1].min(), nz[0].min(), nz[1].max() + 1, nz[0].max() + 1))
    t.save(os.path.join(OUT, f"{name}.png"))
print("tiles written")
