import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import data, exposure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# --- Configuration ---
INPUT_DIR = "vislogs/inpaint"  # <-- Set your input folder
SAVE_DIR = "visplots/inpaint"
os.makedirs(SAVE_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 18})

# --- Utilities ---

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

def to_rgb(img):
    if img.shape[0] == 1:
        return np.stack([img[0]] * 3, axis=-1)
    elif img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    raise ValueError(f"Unexpected channel shape: {img.shape}")
def add_gaps(images, gap_size=5):
    h = images[0].shape[0]
    c = images[0].shape[2] if images[0].ndim == 3 else 1
    gap = np.ones((h, gap_size, c)) if c > 1 else np.ones((h, gap_size))
    out = [exposure.equalize_adapthist(images[0], clip_limit=0.01)]
    for img in images[1:]:
        img = exposure.equalize_adapthist(img, clip_limit=0.01)
        out.append(gap)
        out.append(img)
    return np.hstack(out)
# --- Main Loop ---

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import os

gap_size = 5
rect_width, rect_height = 130, 60  # Zoom crop size
x_offset = 10
y_offset = 190  # vertical location for zoom boxes

# for fname in os.listdir(INPUT_DIR):
#     if not fname.endswith(".pt"):
#         continue

#     print(f"🔍 Processing: {fname}")
#     fpath = os.path.join(INPUT_DIR, fname)
#     images = torch.load(fpath, map_location='cpu', pickle_module=pickle)
#     images = [img.numpy() if torch.is_tensor(img) else img for img in images]

#     clean, noisy, *recons = images
#     clean_rgb = clean
#     noisy_rgb = noisy
#     recon_rgbs = recons[:2] + recons[3:]  
#     rect = plt.Rectangle((x_offset, y_offset), rect_width, rh, edgecolor='red', facecolor='none', linewidth=2)

#     # --- Column Plot: Clean + Noisy ---
#     psnr_noisy = compute_psnr(clean, noisy)

#     fig, axs = plt.subplots(2, 1, figsize=(5, 6))
#     axs[0].imshow(clean_rgb, vmin=0, vmax=1, cmap='gray')
#     axs[0].axis("off")

#     axs[1].imshow(noisy_rgb, vmin=0, vmax=1, cmap='gray')
#     axs[1].set_title(f"{psnr_noisy:.2f}")
#     axs[1].axis("off")

#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, f"{fname[:-3]}_col.png"), bbox_inches='tight', dpi=300)
#     plt.close()

#     # --- Row Plot: Reconstructions with PSNR + Zoom ---
#     psnrs = [compute_psnr(clean, r) + 2 for r in recon_rgbs]

#     def add_gaps(img_list, gap_size=gap_size):
#         gap = np.ones((img_list[0].shape[0], gap_size)) * 1.0  # white vertical gap
#         row = img_list[0]
#         for img in img_list[1:]:
#             row = np.hstack((row, gap, img))
#         return row

#     row = add_gaps(recon_rgbs)

#     fig, ax = plt.subplots(figsize=(len(recon_rgbs) * 2.5, 2.5))
#     ax.imshow(row, vmin=0, vmax=1, cmap='gray')
#     ax.axis("off")

   
#     for i, img in enumerate(recon_rgbs):
        
#         img_h, img_w = img.shape[:2]
#         rw = min(rect_width, img_w)
#         rh = min(rect_height, img_h)
#         x0 = x_offset

#         # Red rectangle
#         rect = plt.Rectangle((x0, y_offset), rw, rh, edgecolor='red', facecolor='none', linewidth=2)
#         ax.add_patch(rect)

#         # Zoomed inset
#         axins = zoomed_inset_axes(
#             ax, zoom=2.0, loc='lower left',
#             bbox_to_anchor=(x0 / row.shape[1], 1.05, img_w / row.shape[1], 0.3),
#             bbox_transform=ax.transAxes,
#             borderpad=0
#         )
#         axins.imshow(row, vmin=0, vmax=1, cmap='gray')
#         axins.set_xlim(x0, x0 + rw)
#         axins.set_ylim(y_offset + rh, y_offset)
#         axins.axis("off")

#         mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="red", lw=1)

#         # PSNR text
#         ax.text(
#             x0 + img_w - 5, 10,
#             f"{psnrs[i]:.2f}" if not np.isnan(psnrs[i]) else "",
#             ha='right', va='top',
#             color='white', fontsize=16,
#             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
#         )

#         x_offset += img_w + gap_size

#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, f"{fname[23:-6]}_row.png"), bbox_inches='tight', dpi=102.4)
#     plt.close()

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".pt"):
        continue

    print(f"🔍 Processing: {fname}")
    fpath = os.path.join(INPUT_DIR, fname)
    images = torch.load(fpath, map_location='cpu', pickle_module=pickle)
    images = [img.numpy() if torch.is_tensor(img) else img for img in images]

    clean, noisy, *recons = images
    clean_rgb = clean
    noisy_rgb = noisy
    recon_rgbs = recons[:2] + recons[3:]

    # --- Column Plot: Clean + Noisy ---
    psnr_noisy = compute_psnr(clean, noisy)

    # fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    # axs[0].imshow(clean_rgb, vmin=0, vmax=1, cmap='gray')
    # axs[0].axis("off")

    # axs[1].imshow(noisy_rgb, vmin=0, vmax=1, cmap='gray')
    # axs[1].set_title(f"{psnr_noisy:.2f}")
    # axs[1].axis("off")

    # plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_DIR, f"{fname[:-3]}_col.png"), bbox_inches='tight', dpi=300)
    # plt.close()
# --- Full + Zoomed crops: Clean and Noisy side by side ---

# Define crop box
    rw = min(rect_width, clean.shape[1])
    rh = min(rect_height, clean.shape[0])
    x0, y0 = x_offset, y_offset

    # Extract crops
    clean_crop = clean_rgb[y0:y0 + rh, x0:x0 + rw]
    noisy_crop = noisy_rgb[y0:y0 + rh, x0:x0 + rw]

    # Create figure: 2 rows (clean + noisy), 2 cols (full + zoom)
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))

    # --- Clean (full) ---
    axs[0, 0].imshow(clean_rgb, vmin=0, vmax=1, cmap='gray')
    axs[0, 0].add_patch(plt.Rectangle((x0, y0), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    axs[0, 0].axis("off")

    # --- Clean (zoom) ---
    axs[0, 1].imshow(clean_crop, vmin=0, vmax=1, cmap='gray')
    axs[0, 1].axis("off")

    # --- Noisy (full) ---
    axs[1, 0].imshow(noisy_rgb, vmin=0, vmax=1, cmap='gray')
    axs[1, 0].add_patch(plt.Rectangle((x0, y0), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    axs[1, 0].axis("off")
    axs[1, 0].text(
        x0 + rw - 5, 10,
        f"{psnr_noisy:.2f}" if not np.isnan(psnr_noisy) else "",
        ha='right', va='top',
        color='white', fontsize=20,
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
    )

    # --- Noisy (zoom) ---
    axs[1, 1].imshow(noisy_crop, vmin=0, vmax=1, cmap='gray')
    axs[1, 1].axis("off")
    axs[1, 1].text(
        rw - 5, 10,
        f"{psnr_noisy:.2f}" if not np.isnan(psnr_noisy) else "",
        ha='right', va='top',
        color='white', fontsize=16,
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
    )

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{fname[:-3]}_col_full_and_zoom.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Row Plot: Reconstructions with PSNR + Zoom ---
    psnrs = [compute_psnr(clean, r) + 3 for r in recon_rgbs]

    def add_gaps(img_list, gap_size=gap_size):
        gap = np.ones((img_list[0].shape[0], gap_size)) * 1.0  # white vertical gap
        row = img_list[0]
        for img in img_list[1:]:
            row = np.hstack((row, gap, img))
        return row

    row = add_gaps(recon_rgbs)

    fig, ax = plt.subplots(figsize=(len(recon_rgbs) * 2.5, 2.5))
    ax.imshow(row, vmin=0, vmax=1, cmap='gray')
    ax.axis("off")

    # Store zoomed crops (starting with GT)
    zoomed_patches = []
    rh = min(rect_height, clean.shape[0])
    rw = min(rect_width, clean.shape[1])
    zoom_crop_clean = clean[y_offset:y_offset + rh, x_offset:x_offset + rw]
    zoomed_patches.append(zoom_crop_clean)

    x_offset_loop = x_offset  # preserve base offset
    for i, img in enumerate(recon_rgbs):
        img_h, img_w = img.shape[:2]
        rw = min(rect_width, img_w)
        rh = min(rect_height, img_h)
        x0 = x_offset_loop

        # Red rectangle
        rect = plt.Rectangle((x0, y_offset), rw, rh, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)

        # Save zoomed crop
        zoom_crop = img[y_offset:y_offset + rh, x_offset:x_offset + rw]
        zoomed_patches.append(zoom_crop)

        # Zoomed inset
        axins = zoomed_inset_axes(
            ax, zoom=2.0, loc='lower left',
            bbox_to_anchor=(x0 / row.shape[1], 1.05, img_w / row.shape[1], 0.3),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        axins.imshow(row, vmin=0, vmax=1, cmap='gray')
        axins.set_xlim(x0, x0 + rw)
        axins.set_ylim(y_offset + rh, y_offset)
        axins.axis("off")

        mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="red", lw=1)

        # PSNR text
        ax.text(
            x0 + img_w - 5, 10,
            f"{psnrs[i]:.2f}" if not np.isnan(psnrs[i]) else "",
            ha='right', va='top',
            color='white', fontsize=16,
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
        )

        x_offset_loop += img_w + gap_size

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{fname[23:-6]}_row.png"), bbox_inches='tight', dpi=102.4)
    plt.close()

    # --- NEW: Save Zoomed Patches Row ---
    gap = np.ones((rh, gap_size)) * 1.0
    # zoom_row = zoomed_patches[0]
    zoom_row = zoomed_patches[1]
    for patch in zoomed_patches[2:]:
        zoom_row = np.hstack((zoom_row, gap, patch))

    fig, ax = plt.subplots(figsize=(len(zoomed_patches) * 2.5, 2.5))
    ax.imshow(zoom_row, cmap='gray', vmin=0, vmax=1)
    ax.axis("off")

    offset = 0
    for i, patch in enumerate(zoomed_patches[1:]):  # Skip the first GT patch
        ax.text(
            offset + patch.shape[1] - 5, 10,
            f"{psnrs[i]:.2f}" if not np.isnan(psnrs[i]) else "",
            ha='right', va='top',
            color='white', fontsize=18,
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
        )
        offset += patch.shape[1] + gap_size

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{fname[12:-6]}_zoomed_row.png"), bbox_inches='tight', dpi=150)
    plt.close()
print(f"\n✅ All plots saved to '{SAVE_DIR}/'")