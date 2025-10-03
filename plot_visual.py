import os
from matplotlib.pylab import gamma
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from skimage import data, exposure

# Directories
LOGS_DIR = "vislogs/ICNN"
PLOTS_DIR = "visplots/ICNN"

# Configuration
train_idx = 2
val_idx = 0
PLOT_INDIVIDUAL = False  # <--- Toggle individual image plots here

# Create output directory
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 20})

# --- Utility Functions ---

def compute_psnr(original, reconstructed):
    mse = np.mean((np.array(original) - np.array(reconstructed)) ** 2, axis=(1, 2))
    mse = np.mean(mse)  # Avg across channels
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def plot(image, title, filename):
    plt.figure(figsize=(10, 6))
    if image.shape[0] == 1:
        image = image[0]
    elif image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image, vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def to_rgb(img_tensor):
    if img_tensor.shape[0] == 1:
        img = img_tensor[0]
        img = np.stack([img]*3, axis=-1)
    elif img_tensor.shape[0] == 3:
        img = np.transpose(img_tensor, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported channel shape: {img_tensor.shape}")
    return np.clip(img, 0, 1)

def is_plottable(value):
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)

# --- Main Loop ---

for filename in os.listdir(LOGS_DIR):
    if filename.endswith(".pt"):
        print(f"\nProcessing {filename}")
        log_path = os.path.join(LOGS_DIR, filename)
        log_data = torch.load(log_path, map_location='cpu', pickle_module=pickle)
        image_train = log_data["train"]
        image_val = log_data["val"]
        checkpoint_cost = log_data["cost"]
        print(f"Train data type: {image_train[0].shape}, Val data type: {image_val[0].shape}")

        original_train = image_train[0]
        original_val = image_val[0]
        noisy_train = image_train[1]
        noisy_val = image_val[1]

        print(f"Train shape: {original_train.shape}, Val shape: {original_val.shape}")
        # Optional: plot individual images
        if PLOT_INDIVIDUAL:
            # plot(original_train[train_idx], "Original Train", f"original_train_{filename[:-3]}.png")
            plot(original_val[val_idx], "Original Val", f"original_val_{filename[:-3]}.png")

            # psnr_noisy_train = compute_psnr(original_train[train_idx], noisy_train[train_idx])
            psnr_noisy_val = compute_psnr(original_val[val_idx], noisy_val[val_idx])
            # plot(noisy_train[train_idx], f"Noisy PSNR: {psnr_noisy_train:.2f}", f"noisy_train_{filename[:-3]}.png")
            plot(noisy_val[val_idx], f"Noisy PSNR: {psnr_noisy_val:.2f}", f"noisy_val_{filename[:-3]}.png")

        # Prepare checkpoint images
        checkpoint_train_imgs = []
        checkpoint_val_imgs = []
        
        psnr_val_list = []
        psnr_train_list = []
        
        # navigate throught image_train[2:], image_val[2:] and delete all the elements up to the last part that starts with Val PSNR inf
        for idx,(img_train_batch, img_val_batch) in enumerate(zip(image_train, image_val)):
            psnr_val_list.append(compute_psnr(original_val[val_idx], img_val_batch[val_idx]))
            psnr_train_list.append(compute_psnr(original_train[train_idx], img_train_batch[train_idx]))
        # Find the last index where PSNR is inf
        inf_val_indices = [i for i, psnr in enumerate(psnr_val_list) if psnr == float('inf')]
        last_valid_idx = inf_val_indices[-1]
        print(f"Last valid index for PSNR: {last_valid_idx}", "size of PSNR list:", len(psnr_val_list))
        # Truncate the lists to only include valid indices

        image_train = image_train[last_valid_idx: len(psnr_val_list)]
        image_val = image_val[last_valid_idx: len(psnr_val_list)]
        save_log = True
        if save_log:
            # overwrite log_data with the truncated lists
            log_data["train"] = image_train
            log_data["val"] = image_val
            log_data["cost"] = checkpoint_cost[last_valid_idx: len(psnr_val_list)]
            # Save the modified log data
            torch.save(log_data, log_path)
            print(f"Truncated train and val data to {len(image_train)} checkpoints.")
        original_train = image_train[0]
        original_val = image_val[0]
        noisy_train = image_train[1]
        # checkpoint_cost = checkpoint_cost[last_valid_idx: len(psnr_val_list)]
        row_train_psnr = []
        row_val_psnr = []
        for idx, (img_train_batch, img_val_batch) in enumerate(zip(image_train[2:], image_val[2:])):
            # if idx == 1 or idx == 2 or idx == 3 or idx == 4:
            #     continue
            psnr_train = compute_psnr(original_train[train_idx], img_train_batch[train_idx])
            psnr_val = compute_psnr(original_val[val_idx], img_val_batch[val_idx])
            row_train_psnr.append(psnr_train)
            row_val_psnr.append(psnr_val)
            print(f"Checkpoint {idx} | Train PSNR: {psnr_train:.2f} | Val PSNR: {psnr_val:.2f}")
            print("cost:", checkpoint_cost)
            if PLOT_INDIVIDUAL:
                plot(img_train_batch[train_idx], f"PSNR: {psnr_train:.2f}, cost {checkpoint_cost[idx]}", f"train{checkpoint_cost[idx]}_{filename[:-3]}.png")
                plot(img_val_batch[val_idx], f"PSNR: {psnr_val:.2f}, cost {checkpoint_cost[idx]}", f"val{checkpoint_cost[idx]}_{filename[:-3]}.png")

            checkpoint_train_imgs.append(to_rgb(img_train_batch[train_idx]))
            checkpoint_val_imgs.append(to_rgb(img_val_batch[val_idx]))
        def enhance_contrast(img, clip_percent=0):
            img = np.copy(img)
            if img.ndim == 3:  # Color image
                for ch in range(img.shape[2]):
                    p_low, p_high = np.percentile(img[:, :, ch], [clip_percent, 100 - clip_percent])
                    img[:, :, ch] = np.clip((img[:, :, ch] - p_low) / (p_high - p_low + 1e-8), 0, 1)
            else:  # Grayscale
                p_low, p_high = np.percentile(img, [clip_percent, 100 - clip_percent])
                img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
            return img
        def gamma_correction(img, gamma=0.8):
            return np.clip(img ** gamma, 0, 1)
        
        def add_gaps(images, gap_size=5):
            h = images[0].shape[0]
            c = images[0].shape[2] if images[0].ndim == 3 else 1
            gap = np.ones((h, gap_size, c)) if c > 1 else np.ones((h, gap_size))
            out = [exposure.equalize_adapthist(images[0], clip_limit=0.02)]
            for img in images[1:]:
                img = exposure.equalize_adapthist(img, clip_limit=0.02)
                out.append(gap)
                out.append(img)
            return np.hstack(out)
          # Optionally plot GT and Noisy images separately
        PLOT_GT_NOISY_SEPARATELY = True  # <--- Toggle this option as needed
        if PLOT_GT_NOISY_SEPARATELY:
            psnr_noisy_val = compute_psnr(original_val[val_idx], noisy_val[val_idx])
            fig, axes = plt.subplots(2, 1, figsize=(2.5, 5))
            # Val GT
            img_gt = original_val[val_idx]
            # img_gt = enhance_contrast(img_gt, clip_percent=1.5)
            if img_gt.shape[0] == 1:
                img_gt = img_gt[0]
            elif img_gt.shape[0] == 3:
                img_gt = np.transpose(img_gt, (1, 2, 0))
            img_gt = np.clip(img_gt, 0, 1)
            img_gt = exposure.equalize_adapthist(img_gt, clip_limit=0.02)
            axes[0].imshow(img_gt, vmin=0, vmax=1)
            # axes[0].set_title("Val GT")
            axes[0].axis('off')
            # Val Noisy
            # clamp the noisy image to [0, 1] range
            img_noisy = noisy_val[val_idx]
            # img_noisy = enhance_contrast(img_noisy, clip_percent=1.5)
            img_noisy = np.clip(img_noisy, 0, 1)
            if img_noisy.shape[0] == 1:
                img_noisy = img_noisy[0]
            elif img_noisy.shape[0] == 3:
                img_noisy = np.transpose(img_noisy, (1, 2, 0))
            img_noisy = exposure.equalize_adapthist(img_noisy, clip_limit=0.02)
            # img_noisy = img_gt + np.random.randn(*img_noisy.shape) * 0.15
            axes[1].imshow(img_noisy, vmin=0, vmax=1)
            axes[1].set_title(f"{psnr_noisy_val:.2f}")
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"val_gt_noisy_col_{filename[:-3]}.png"), bbox_inches='tight', dpi = 100)
            plt.close()
            row_val_imgs = checkpoint_val_imgs
        else:
            # Combine GT and Noisy images with checkpoints
            row_val_imgs = [to_rgb(original_val[val_idx]), to_rgb(noisy_val[val_idx])] + checkpoint_val_imgs

        # Create and save row plots
        row_train_imgs = [to_rgb(original_train[train_idx]), to_rgb(noisy_train[train_idx])] + checkpoint_train_imgs
        # Train row
        row_train = np.hstack(row_train_imgs)
  

        # Val row
        # Add a small gap (e.g., 5 pixels of white) between images in the row
        row_val = np.hstack(row_val_imgs)
       
        # row_val_imgs = [gamma_correction(enhance_contrast(img, clip_percent=1.5), gamma=1) for img in row_val_imgs]
        # row_val = add_gaps(row_val_imgs)
        # plt.figure(figsize=(len(row_val_imgs) * 2.5, 2.5))
        # plt.imshow(row_val, interpolation='none', vmin=0, vmax=1)
        # # Draw an 80x20 rectangle on each image in the row
        # x_offset = 0
        # for img in row_val_imgs:
        #     img_width = img.shape[1]
        #     img_height = img.shape[0]
        #     rect_x = x_offset
        #     rect_y = 0
        #     rect_width = min(80, img_width)
        #     rect_height = min(20, img_height)
        #     rectangle = plt.Rectangle(
        #     (rect_x, rect_y), rect_width, rect_height,
        #     edgecolor='red', facecolor='none', linewidth=2
        #     )
        #     plt.gca().add_patch(rectangle)
        #     x_offset += img_width + 5  # 5 is the gap size used in add_gaps
        #     zoomed_rec = zoomed_inset_axes(plt.gca(), zoom=1.5, loc='upper left', borderpad=2)
        #     zoomed_rec.imshow(img, interpolation='none', vmin=0, vmax=1)
        #     zoomed_rec.set_xlim(80, img_width)
        #     zoomed_rec.set_ylim(img_height, 20)  # Invert y-axis for 'upper' origin
        #     zoomed_rec.axis('off')
        #     mark_inset(plt.gca(), zoomed_rec, loc1=2, loc2=1, fc="none", ec="red", lw=1)
        # for i, psnr in enumerate(row_val_psnr):
        #     img_width = row_val_imgs[i].shape[1]
        #     x0 = sum(img.shape[1] for img in row_val_imgs[:i])  # Left edge of current patch
        #     x = x0 + img_width - 5  # 5 pixels from right edge
        #     y = 5  # 5 pixels from top edge

        #     plt.text(
        #         x, y,
        #         f"{psnr:.2f}" if not np.isnan(psnr) else "",
        #         ha='right', va='top',  # right-align for top-right corner
        #         color='white', fontsize=20,
        #         bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
        #     )
        # plt.tight_layout()
        # plt.axis("off")
        # plt.savefig(os.path.join(PLOTS_DIR, f"row_val_{filename[:-3]}.png"), bbox_inches="tight", dpi=76.8)
        # plt.close()

        # --- Add gaps between images ---
        # --- Add gaps between images ---
        row_val = add_gaps(row_val_imgs)

        # --- Setup figure and axis ---
        fig, ax = plt.subplots(figsize=(len(row_val_imgs) * 2.5, 2.5))
        ax.imshow(row_val, interpolation='none', vmin=0, vmax=1)
        ax.axis("off")

        # Parameters
        gap_size = 5
        rect_width, rect_height = 70, 30  # Zoom rectangle size

        # --- Draw red rectangles and add zoomed insets ---
        x_offset = 50
        y_offset = 50  # Starting y position for rectangles
        for i, img in enumerate(row_val_imgs):
            img_h, img_w = img.shape[:2]

            # Position of the current image in the full row_val image
            x0 = x_offset
            x_offset += img_w + gap_size

            # Rectangle coordinates (on the full row image)
            rw = min(rect_width, img_w)
            rh = min(rect_height, img_h)
            rect = plt.Rectangle((x0, y_offset), rw, rh, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

            # --- Create zoomed inset from row_val ---
            axins = zoomed_inset_axes(
                ax, zoom=2.0, loc='lower left',
                bbox_to_anchor=(x0 / row_val.shape[1], 1.05, img_w / row_val.shape[1], 0.3),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            axins.imshow(row_val, interpolation='none', vmin=0, vmax=1)
            axins.set_xlim(x0, x0 + rw)
            axins.set_ylim(rh+y_offset, y_offset)  # invert y
            axins.axis("off")

            # Draw connecting lines from red box to inset
            mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="red", lw=1)

        # --- Overlay PSNR values ---
        x_offset = 0
        for i, psnr in enumerate(row_val_psnr):
            img_w = row_val_imgs[i].shape[1]
            x = x_offset + img_w - 5
            y = 5
            x_offset += img_w + gap_size

            ax.text(
                x, y,
                f"{psnr:.2f}" if not np.isnan(psnr) else "",
                ha='right', va='top',
                color='white', fontsize=20,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
            )

        # --- Save plot ---
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"row_val_{filename[:-3]}.png"), bbox_inches="tight", dpi=100)
        plt.close()

print(f"\n✅ All plots saved to '{PLOTS_DIR}/'")