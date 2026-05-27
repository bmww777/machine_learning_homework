"""
Semantic Segmentation — Batch Inference & Visualization Demo
=============================================================
Two modes:
  1. Default: load ALL .pth files, run per-model inference (3-col grids).
  2. Compare (--compare): load BCE / Dice / BCEDice (all ResNet34), run on the
     SAME 5 defect images, generate a 5-column composite figure that makes
     gradient collapse (BCE), boundary jaggedness (Dice), and smooth continuity
     (BCEDice) directly comparable.

Outputs are saved to: results/

Usage:
    python demo_predict.py                          # Default: all models, 3-col grid
    python demo_predict.py --compare                # Loss comparison: 5-col grid
    python demo_predict.py --compare --num_samples 5
    python demo_predict.py --checkpoint x.pth       # Single model only
"""

import os
import sys
import random
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import UNetMultiBackbone
from dataset import get_val_transforms, rle2mask


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Default: all .pth files
CHECKPOINT_GLOB_PATTERNS = [
    'ablation_*.pth',
    'best_steel_unet.pth',
    'best_*.pth',
    '*.pth',
]

# Compare mode: three ResNet34 models (fair loss-function comparison)
COMPARE_CHECKPOINTS = {
    'BCE':     'ablation_E_resnet34_bce.pth',
    'Dice':    'ablation_D_resnet34_dice.pth',
    'BCEDice': 'ablation_C_resnet34_bcedice.pth',
}

IMAGE_DIR = 'data/severstal/train_images'
CSV_PATH = 'data/severstal/train.csv'
DEMO_DIR = 'demo_samples'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'results'


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def find_all_checkpoints():
    """Discover all available .pth model checkpoints in the project directory.

    Deduplicates by absolute path to avoid the same file appearing under
    multiple glob patterns.

    Returns:
        list[str]: Sorted list of absolute checkpoint paths.

    Raises:
        FileNotFoundError: if no .pth files are found at all.
    """
    seen = set()
    checkpoints = []

    for pattern in CHECKPOINT_GLOB_PATTERNS:
        for ckpt in glob(pattern):
            abspath = os.path.abspath(ckpt)
            if abspath not in seen:
                seen.add(abspath)
                checkpoints.append(abspath)

    if not checkpoints:
        raise FileNotFoundError(
            "No .pth checkpoint files found in the project directory.\n"
            "Please run train.py first to generate at least one model weight.\n"
            f"Scanned patterns: {CHECKPOINT_GLOB_PATTERNS}"
        )

    checkpoints.sort()
    print(f"  Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        print(f"    - {os.path.basename(ckpt)}")
    return checkpoints


def load_model_from_checkpoint(checkpoint_path, device):
    """Restore model architecture & weights from a saved checkpoint.

    Reads backbone name and num_classes from checkpoint config to
    reconstruct the exact same UNetMultiBackbone structure used during training.

    Args:
        checkpoint_path: path to .pth file.
        device         : torch.device.

    Returns:
        tuple: (model, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_config = ckpt.get('config', {})
    backbone_name = saved_config.get('backbone', 'resnet34')

    model = UNetMultiBackbone(
        backbone_name=backbone_name,
        pretrained=False,
        num_classes=saved_config.get('num_classes', 1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    best_dice = ckpt.get('val_dice', None)
    if best_dice is not None:
        print(f"    backbone={backbone_name}, saved val_dice={best_dice:.4f}")
    else:
        print(f"    backbone={backbone_name}")

    return model, ckpt


def find_sample_images(num_samples=3, csv_path=None):
    """Determine which images to use for inference.

    Priority:
      1. demo_samples/  directory (if exists and non-empty).
      2. data/severstal/train_images/  — only images that have GT entries
         in the CSV (preferring defect-bearing images).

    Args:
        num_samples: number of images to return.
        csv_path   : path to train.csv, used to filter by GT availability.

    Returns:
        tuple: (image_paths, ground_truth_available)
    """
    demo_dir = Path(DEMO_DIR)
    if demo_dir.is_dir():
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        paths = []
        for ext in exts:
            paths.extend(glob(str(demo_dir / ext)))
        if paths:
            print(f"\n  Using {len(paths)} image(s) from demo_samples/")
            return sorted(paths)[:num_samples], False

    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise FileNotFoundError(
            "No test images found. Place images in one of:\n"
            f"  • {DEMO_DIR}/  (recommended: a few .jpg files)\n"
            f"  • {IMAGE_DIR}/ (Kaggle dataset)"
        )

    all_images = glob(str(image_dir / '*.jpg')) + glob(str(image_dir / '*.jpeg'))
    if not all_images:
        raise FileNotFoundError(f"No .jpg/.jpeg images found in {IMAGE_DIR}")

    # ── Cross-reference with CSV to prefer images that have GT ──
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

        if df is not None and 'ImageId' in df.columns and 'EncodedPixels' in df.columns:
            # Strip file extension from CSV ImageId (some CSVs include ".jpg")
            df['_clean_id'] = df['ImageId'].astype(str).str.rsplit('.', n=1).str[0]

            # Build sets using cleaned IDs
            gt_ids = set(df['_clean_id'].unique())

            has_defect_ids = set(
                df[df['EncodedPixels'].notna() &
                   (df['EncodedPixels'] != '') &
                   (df['EncodedPixels'] != '-1')]['_clean_id'].unique()
            )
            no_defect_ids = gt_ids - has_defect_ids

            # Among all images on disk, find those present in CSV
            disk_ids = {Path(p).stem for p in all_images}
            overlap = gt_ids & disk_ids

            # ── Always print diagnostic: sample IDs from both sides ──
            csv_samples = list(sorted(gt_ids)[:4])
            disk_samples = list(sorted(disk_ids)[:4])
            print(f"\n  CSV unique ImageIds : {len(gt_ids)}")
            print(f"  Disk image files    : {len(disk_ids)}")
            print(f"  Overlap (matched)   : {len(overlap)}")
            print(f"  CSV samples         : {csv_samples}")
            print(f"  Disk samples        : {disk_samples}")

            gt_on_disk_with_defect = [p for p in all_images
                                      if Path(p).stem in has_defect_ids & disk_ids]
            gt_on_disk_clean = [p for p in all_images
                                if Path(p).stem in no_defect_ids & disk_ids]

            # Candidate pool: only defect-bearing images
            candidate_pool = gt_on_disk_with_defect
            if len(candidate_pool) < num_samples:
                print(f"\n  Only {len(candidate_pool)} defect images available "
                      f"(need {num_samples}), adding clean plates...")
                candidate_pool += gt_on_disk_clean

            if candidate_pool:
                selected = random.sample(
                    candidate_pool,
                    min(num_samples, len(candidate_pool))
                )
                n_with = sum(1 for p in selected if Path(p).stem in has_defect_ids)
                n_clean = sum(1 for p in selected if Path(p).stem in no_defect_ids)
                n_nogt = sum(1 for p in selected if Path(p).stem not in gt_ids)
                print(f"\n  Sampled {len(selected)} image(s) from train_images/:")
                print(f"    with defects:  {n_with}")
                print(f"    clean plates:  {n_clean}")
                print(f"    no CSV entry:  {n_nogt}")
                return selected, True

    # Fallback: pure random without CSV filtering
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\n  Using {len(selected)} random image(s) from train_images/ (no CSV filter)")
    return selected, os.path.exists(csv_path) if csv_path else False


def load_ground_truth(image_path, csv_path, verbose=True):
    """Attempt to load ground-truth mask from Kaggle CSV.

    Distinguishes three cases:
      - CSV unavailable / image not in CSV  →  returns None  (no GT at all)
      - Image in CSV but has zero defects   →  returns all-zero ndarray (valid GT)
      - Image has defects                   →  returns binary mask

    Args:
        image_path: image file path (used to derive ImageId).
        csv_path  : path to train.csv.
        verbose   : print diagnostic info about each lookup step.

    Returns:
        np.ndarray: (256, 1600) binary mask (may be all-zero).
        None      : if CSV or image entry genuinely unavailable.
    """
    if not os.path.exists(csv_path):
        if verbose:
            print(f"      [GT] CSV not found: {csv_path}")
        return None

    image_id = Path(image_path).stem

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        if verbose:
            print(f"      [GT] Failed to read CSV: {e}")
        return None

    if 'ImageId' not in df.columns or 'EncodedPixels' not in df.columns:
        if verbose:
            print(f"      [GT] CSV columns: {list(df.columns)} (need ImageId + EncodedPixels)")
        return None

    # Normalise: strip file extension from CSV ImageId (some CSVs include ".jpg")
    df['_image_id_clean'] = df['ImageId'].astype(str).str.rsplit('.', n=1).str[0]
    records = df[df['_image_id_clean'] == image_id]
    if len(records) == 0:
        if verbose:
            # Show a few IDs from CSV to help spot naming mismatches
            sample_ids = df['ImageId'].unique()[:5]
            print(f"      [GT] image_id='{image_id}' not found in CSV.")
            print(f"           CSV sample IDs: {list(sample_ids)} ...")
        return None

    mask = np.zeros((256, 1600), dtype=np.uint8)
    defect_count = 0
    for _, row in records.iterrows():
        rle = row['EncodedPixels']
        if pd.isna(rle) or str(rle).strip() in ('', '-1'):
            continue
        class_mask = rle2mask(str(rle), shape=(256, 1600))
        mask = np.maximum(mask, class_mask)
        defect_count += 1

    if verbose:
        total_defect_pixels = int(mask.sum())
        if total_defect_pixels > 0:
            print(f"      [GT] {defect_count} defect class(es), "
                  f"{total_defect_pixels} defect pixels "
                  f"({total_defect_pixels / (256 * 1600) * 100:.2f}%)")
        else:
            print(f"      [GT] image found in CSV, 0 defect pixels (clean plate)")

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, image_bgr, device, threshold=0.5):
    """Run forward pass on a single image and return a binary prediction mask.

    Args:
        model     : UNetMultiBackbone in eval mode.
        image_bgr : (H, W, 3) BGR image, dtype=np.uint8.
        device    : torch.device.
        threshold : binarization threshold, default 0.5.

    Returns:
        np.ndarray: (H, W) binary mask, dtype=np.uint8.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transforms = get_val_transforms((256, 1600))
    augmented = transforms(image=image_rgb)
    image_tensor = augmented['image']  # (3, 256, 1600) normalized

    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, 256, 1600)

    logits = model(image_tensor)
    probs = torch.sigmoid(logits)
    pred_binary = (probs > threshold).float().squeeze().cpu().numpy()

    return pred_binary.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def draw_overlay(image_rgb, mask, color=(1.0, 0.0, 0.0), alpha=0.45):
    """Superimpose a coloured semi-transparent mask on an RGB image.

    Args:
        image_rgb: (H, W, 3) float RGB image, values in [0, 1].
        mask     : (H, W) binary mask, dtype=np.uint8.
        color    : overlay colour (R, G, B), default red (1, 0, 0).
        alpha    : blend strength, default 0.45.

    Returns:
        np.ndarray: (H, W, 3) blended image, clipped to [0, 1].
    """
    output = image_rgb.copy()
    coloured = np.zeros_like(image_rgb)
    for c in range(3):
        coloured[:, :, c] = color[c]
    coloured *= mask[:, :, np.newaxis]

    output = (1 - alpha * mask[:, :, np.newaxis]) * output + alpha * coloured
    return np.clip(output, 0, 1)


def create_demo_figure(samples, model_label, save_path):
    """Generate a 3-column × N-row comparison grid and save to disk.

    Column layout per sample row:
      [Input Image] | [Ground Truth Mask] | [Prediction Overlay]

    Args:
        samples    : list of dicts with keys 'image_bgr', 'gt_mask',
                     'pred_mask', 'image_id'.
        model_label: human-readable model name for the figure title.
        save_path  : output .png path.
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 1.1 * num_samples))
    plt.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.04,
                        wspace=0.015, hspace=0.04)

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['(a) Input', '(b) GT', '(c) Pred Overlay']

    for row_idx, sample in enumerate(samples):
        image_bgr = sample['image_bgr']
        gt_mask = sample.get('gt_mask', None)
        pred_mask = sample['pred_mask']
        image_id = sample.get('image_id', f'Sample {row_idx + 1}')

        # Convert to float RGB [0, 1]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = cv2.resize(image_rgb, (1600, 256))

        # ── Column 1: Input Image ──
        axes[row_idx, 0].imshow(image_rgb)
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=9, fontweight='bold', pad=6)
        axes[row_idx, 0].axis('off')

        # ── Column 2: Ground Truth ──
        ax_gt = axes[row_idx, 1]
        if gt_mask is not None:
            ax_gt.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        else:
            ax_gt.text(0.5, 0.5, 'No GT', transform=ax_gt.transAxes,
                       ha='center', va='center', fontsize=8, color='gray')
        if row_idx == 0:
            ax_gt.set_title(col_titles[1], fontsize=9, fontweight='bold', pad=6)
        ax_gt.axis('off')

        # ── Column 3: Prediction Overlay ──
        overlay = draw_overlay(image_rgb, pred_mask.astype(np.uint8),
                               color=(1.0, 0.0, 0.0), alpha=0.45)
        axes[row_idx, 2].imshow(overlay)
        if row_idx == 0:
            axes[row_idx, 2].set_title(col_titles[2], fontsize=9, fontweight='bold', pad=6)
        axes[row_idx, 2].axis('off')

        # Row label
        axes[row_idx, 0].set_ylabel(image_id, fontsize=7, fontweight='bold',
                                    rotation=0, ha='right', va='center', labelpad=8)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.45, label='Predicted Defect'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='Threshold = 0.5'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=8, framealpha=0.8)

    fig.suptitle(f'Model: {model_label}',
                 fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    print(f"  -> saved: {os.path.abspath(save_path)}")


def compute_metrics(pred, gt, smooth=1.0):
    """Compute per-image segmentation metrics.

    All metrics are computed on binary masks (0/1).  None of them are
    directly optimised by BCE, Dice, or BCEDice simultaneously, making
    them fair cross-loss comparison tools.

    Returns:
        dict with keys: dice, iou, precision, recall, fpr
    """
    pred = pred.flatten().astype(np.float64)
    gt = gt.flatten().astype(np.float64)

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    # False Positive Rate: what fraction of predicted defect pixels are wrong
    fpr = (fp + smooth) / (fp + tp + smooth)

    return {
        'dice': round(dice, 4),
        'iou': round(iou, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'fpr': round(fpr, 4),
    }


def create_comparison_figure(image_paths, all_preds, csv_path, save_path):
    """Generate a tight 5-column × N-row loss-function comparison grid.

    Columns per row:
      [Input Image] | [Ground Truth] | [BCE Pred] | [Dice Pred] | [BCEDice Pred]

    All predictions are rendered as red semi-transparent overlays on the
    original image, making boundary quality and false positives directly
    comparable across loss functions.  Spacing is minimised for paper insertion.

    Args:
        image_paths: list of image file paths (N samples).
        all_preds  : dict {loss_name: [pred_mask_N, ...]} for each loss function.
        csv_path   : path to train.csv for ground-truth loading.
        save_path  : output .png path.
    """
    loss_names = list(all_preds.keys())
    n_rows = len(image_paths)
    n_cols = 2 + len(loss_names)

    # Compact figure: images are 1600×256 (6.25:1 aspect), so rows are very flat
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 1.1 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Nearly zero inter-plot gap, leave room above first row
    plt.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.04,
                        wspace=0.01, hspace=0.02)

    col_labels = ['(a) Input', '(b) GT'] + \
                 [f'({chr(99 + i)}) {name}' for i, name in enumerate(loss_names)]

    for row_idx, img_path in enumerate(image_paths):
        image_id = Path(img_path).stem
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_mask = load_ground_truth(img_path, csv_path)

        # ── Column 1: Input ──
        axes[row_idx, 0].imshow(image_rgb)
        axes[row_idx, 0].axis('off')

        # ── Column 2: GT ──
        if gt_mask is not None:
            axes[row_idx, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        else:
            axes[row_idx, 1].text(0.5, 0.5, 'No GT', fontsize=6, color='gray',
                                  transform=axes[row_idx, 1].transAxes,
                                  ha='center', va='center')
        axes[row_idx, 1].axis('off')

        # ── Columns 3-5: Prediction Overlays ──
        for col_offset, loss_name in enumerate(loss_names):
            pred = all_preds[loss_name][row_idx]
            overlay = draw_overlay(image_rgb, pred.astype(np.uint8),
                                   color=(1.0, 0.0, 0.0), alpha=0.45)
            axes[row_idx, 2 + col_offset].imshow(overlay)
            axes[row_idx, 2 + col_offset].axis('off')

            n_defect = int(pred.sum())
            axes[row_idx, 2 + col_offset].text(
                0.98, 0.02, f'{n_defect} px',
                transform=axes[row_idx, 2 + col_offset].transAxes,
                ha='right', va='bottom', fontsize=5, color='white',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

        # Row label — minimal
        axes[row_idx, 0].set_ylabel(image_id, fontsize=6, fontweight='bold',
                                    rotation=0, ha='right', va='center', labelpad=5)

    # Column titles
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=8, fontweight='bold', pad=6)

    # Legend
    legend_els = [
        mpatches.Patch(facecolor='red', alpha=0.45, label='Predicted Defect (Red Overlay)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=1,
               fontsize=7, framealpha=0.8)

    fig.suptitle('Loss Function Comparison — BCE vs Dice vs BCEDice (ResNet34)',
                 fontsize=10, fontweight='bold', y=0.96)
    fig.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.03)
    print(f"  -> saved: {os.path.abspath(save_path)}")
    print(f"  -> saved: {os.path.abspath(save_path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference demo — runs ALL found checkpoints"
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Run a single checkpoint only (default: all found .pth files)')
    parser.add_argument('--compare', action='store_true',
                        help='Loss comparison mode: BCE vs Dice vs BCEDice (ResNet34) on same images')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample images (default: 5)')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to train.csv (overrides default)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    print("=" * 70)
    mode_str = "Loss Comparison Mode" if args.compare else "Batch Inference Demo"
    print(f"Steel Defect Detection — {mode_str}")
    print("=" * 70)

    device = torch.device(DEVICE)
    print(f"\nDevice: {device}")

    csv_path = args.csv if args.csv else CSV_PATH
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # MODE A: Loss Comparison — BCE vs Dice vs BCEDice
    # ══════════════════════════════════════════════════════════════════
    if args.compare:
        N_FIGURES = 10
        SAMPLES_PER_FIGURE = args.num_samples

        print("\n[1/3] Loading three ResNet34 models (BCE / Dice / BCEDice)...")
        models = {}
        for loss_name, ckpt_name in COMPARE_CHECKPOINTS.items():
            if not os.path.exists(ckpt_name):
                print(f"  ERROR: {ckpt_name} not found. Run the ablation experiments first.")
                sys.exit(1)
            model, _ = load_model_from_checkpoint(ckpt_name, device)
            models[loss_name] = model
            params = sum(p.numel() for p in model.parameters())
            print(f"  {loss_name:8s}: {ckpt_name} ({params/1e6:.2f}M)")

        # Pre-load all candidate defect images so we can draw without replacement
        print(f"\n[2/3] Building defect-image pool (CSV: {csv_path})...")
        df = pd.read_csv(csv_path)
        df['_clean'] = df['ImageId'].astype(str).str.rsplit('.', n=1).str[0]
        defect_ids = set(df[df['EncodedPixels'].notna() &
                           (df['EncodedPixels'] != '') &
                           (df['EncodedPixels'] != '-1')]['_clean'].unique())
        all_images = glob(str(Path(IMAGE_DIR) / '*.jpg')) + \
                     glob(str(Path(IMAGE_DIR) / '*.jpeg'))
        pool = [p for p in all_images if Path(p).stem in defect_ids]
        print(f"  Pool: {len(pool)} defect-bearing images available")
        random.shuffle(pool)

        print(f"\n[3/3] Generating {N_FIGURES} comparison figures "
              f"({SAMPLES_PER_FIGURE} samples each, threshold = {args.thresh})...\n")

        # Accumulate metrics across all figures for final summary
        all_metrics = {ln: [] for ln in COMPARE_CHECKPOINTS}  # {loss: [metrics_dict, ...]}

        used = 0
        for fig_idx in range(1, N_FIGURES + 1):
            if used + SAMPLES_PER_FIGURE > len(pool):
                random.shuffle(pool)
                used = 0
            image_paths = pool[used:used + SAMPLES_PER_FIGURE]
            used += SAMPLES_PER_FIGURE

            # Inference + metrics
            all_preds = {}
            fig_metrics = {ln: {} for ln in models}  # per-loss averaged metrics
            for loss_name, model in models.items():
                preds = []
                met_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'fpr': 0}
                for img_path in image_paths:
                    img = cv2.imread(img_path)
                    pred = predict(model, img, device, threshold=args.thresh)
                    preds.append(pred)
                    gt = load_ground_truth(img_path, csv_path)
                    if gt is not None:
                        m = compute_metrics(pred, gt)
                        for k in met_sum:
                            met_sum[k] += m[k]
                all_preds[loss_name] = preds
                n = len(image_paths)
                for k in met_sum:
                    fig_metrics[loss_name][k] = round(met_sum[k] / n, 4)
                    all_metrics[loss_name].append(met_sum[k] / n)

            ids = ' | '.join(Path(p).stem[:12] for p in image_paths)
            print(f"  [{fig_idx:2d}/{N_FIGURES}] {ids}")
            # Print mini metrics table
            header = f"         {'':>10s}" + ''.join(f"{ln:>10s}" for ln in models)
            print(header.format(''))
            for met_name in ['dice', 'iou', 'precision', 'recall', 'fpr']:
                row = f"         {met_name:>10s}" + \
                      ''.join(f"{fig_metrics[ln][met_name]:>10.4f}" for ln in models)
                print(row)

            save_path = output_dir / f'loss_comparison_{fig_idx:02d}.png'
            create_comparison_figure(image_paths, all_preds, csv_path, str(save_path))

        # ── Final summary across all figures ──
        print(f"\n{'=' * 70}")
        print("FINAL SUMMARY — averaged over all figures ± std")
        print(f"{'=' * 70}")
        header = f"{'Metric':>12s}" + ''.join(f"{ln:>14s}" for ln in models)
        print(header)
        print('-' * (12 + 14 * len(models)))
        for met_name in ['dice', 'iou', 'precision', 'recall', 'fpr']:
            vals = [np.array(all_metrics[ln]) for ln in models]
            row = f"{met_name:>12s}" + \
                  ''.join(f"  {v.mean():.4f}±{v.std():.4f}" for v in vals)
            print(row)
        print(f"\n  Key insight (FPR = false positive rate):")
        print(f"    Lower FPR → fewer isolated noise pixels")
        print(f"    Higher Precision → cleaner boundaries, less over-segmentation")
        print(f"    Dice/IoU are reported for reference but biased towards Dice-trained models.")
        print(f"{'=' * 70}\n")

        print(f"{N_FIGURES} figures saved to {output_dir}/:")
        for f in sorted(output_dir.glob('loss_comparison_*.png')):
            print(f"  {f.name}")
        print("=" * 70)
        return

    # ══════════════════════════════════════════════════════════════════
    # MODE B: Standard — one figure per checkpoint
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/5] Searching for model checkpoints...")
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_all_checkpoints()

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────
    # 2. Find sample images (shared across all models)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[2/5] Finding sample images (CSV: {csv_path})...")
    image_paths, has_gt = find_sample_images(args.num_samples, csv_path=csv_path)
    if has_gt and not os.path.exists(csv_path):
        print(f"  WARNING: CSV not found, GT will be unavailable")
    print(f"  Selected {len(image_paths)} image(s) for inference")

    # ── 3. Prepare output directory ──
    # ──────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────
    # 4. Run inference for each checkpoint
    # ──────────────────────────────────────────────────────────────────
    for ckpt_idx, checkpoint_path in enumerate(checkpoints):
        ckpt_name = Path(checkpoint_path).stem  # filename without .pth
        print(f"\n{'=' * 70}")
        print(f"[{ckpt_idx + 1}/{len(checkpoints)}] Model: {ckpt_name}")
        print(f"{'=' * 70}")

        # ── Load model ──
        print("  Loading model...")
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params / 1e6:.2f}M")

        # ── Run inference on all samples ──
        print(f"  Running inference (threshold = {args.thresh})...")
        samples = []
        for idx, img_path in enumerate(image_paths):
            image_id = Path(img_path).stem
            print(f"    [{idx + 1}/{len(image_paths)}] {image_id}")

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"      WARNING: cannot read, skipping")
                continue

            pred_mask = predict(model, image_bgr, device, threshold=args.thresh)

            gt_mask = None
            if has_gt:
                gt_mask = load_ground_truth(img_path, csv_path)

            samples.append({
                'image_bgr': image_bgr,
                'gt_mask': gt_mask,
                'pred_mask': pred_mask,
                'image_id': image_id,
            })

            defect_ratio = pred_mask.sum() / (256 * 1600) * 100
            print(f"      defect pixels: {int(pred_mask.sum())} ({defect_ratio:.3f}%)")

        if not samples:
            print("  ERROR: no valid samples. Skipping this model.")
            continue

        # ── Generate visualisation ──
        model_label = ckpt_name.replace('_', ' ')
        save_path = output_dir / f"{ckpt_name}_results.png"
        print(f"\n  Generating comparison figure...")
        create_demo_figure(samples, model_label, str(save_path))

    # ──────────────────────────────────────────────────────────────────
    # 5. Summary
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Batch inference complete!")
    print(f"  Models processed : {len(checkpoints)}")
    print(f"  Samples per model: {len(image_paths)}")
    print(f"  Output directory : {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
