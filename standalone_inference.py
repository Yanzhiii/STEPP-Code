#!/usr/bin/env python3
"""
Standalone STEPP inference script — no ROS required.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate stepp
    python standalone_inference.py

This script runs the full STEPP inference pipeline on a single image:
    1. DINOv2 feature extraction
    2. SLIC superpixel segmentation
    3. MLP autoencoder reconstruction → traversability cost
    4. Save result images (no GUI window)
"""

import sys
import os
import time
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # MUST be before importing pyplot — no GUI window
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image as PILImage

# Add project root to path (in case not installed via pip)
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from STEPP.DINO.dino_feature_extract import DinoInterface, average_dino_feature_segment_tensor
from STEPP.SLIC.slic_segmentation import SLIC
from STEPP.model.mlp import ReconstructMLP

warnings.filterwarnings("ignore")


def run_inference(
    image_path: str,
    model_path: str,
    mode: str = "segment_wise",
    threshold: float = 0.15,
    output_dir: str = None,
):
    """
    Run STEPP traversability inference on a single image.

    Args:
        image_path:  Path to input RGB image.
        model_path:  Path to trained MLP checkpoint (.pth).
        mode:        'segment_wise' (SLIC superpixels) or 'pixel_wise' (per-pixel).
        threshold:   Traversability cost cap (0.0–1.0). Lower = more conservative.
        output_dir:  Where to save results. Defaults to ./results/stepp_inference/

    Returns:
        fig:         Matplotlib figure handle (for optional inline display in VSCode).
        result_path: Path to the saved result image.
    """
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STEPP] Using device: {device}")

    if output_dir is None:
        output_dir = os.path.join(_PROJ_ROOT, "results", "stepp_inference")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    t_start = time.time()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img_bgr.shape[:2]
    print(f"[STEPP] Image loaded: {W}x{H}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(img_rgb, (H, H))  # square resize (matches original code)
    print(f"[STEPP] Resized to square: {H}x{H}")

    # ------------------------------------------------------------------
    # 2. DINOv2 feature extraction
    # ------------------------------------------------------------------
    print("[STEPP] Loading DINOv2 model (vit_small, first run downloads ~170 MB)...")
    dino = DinoInterface(
        device=device,
        backbone="dinov2",
        input_size=700,
        backbone_type="vit_small",
        patch_size=14,
        interpolate=False,
        use_mixed_precision=False,
    )

    torch_img = torch.from_numpy(small_image).permute(2, 0, 1)
    torch_img = (torch_img.float() / 255.0).unsqueeze(0).to(device)

    print("[STEPP] Extracting DINOv2 features...")
    features = dino.inference(torch_img)  # shape: (1, 384, 50, 50)
    print(f"[STEPP] Feature shape: {features.shape}")

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    # Load MLP model
    print(f"[STEPP] Loading MLP checkpoint: {model_path}")
    model = ReconstructMLP(384, [256, 128, 64, 32, 64, 128, 256])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loss_fn = nn.MSELoss(reduction="none")

    if mode == "segment_wise":
        # ---- 3a. SLIC superpixel segmentation ----
        print("[STEPP] Running SLIC superpixel segmentation (400 segments)...")
        slic = SLIC(crop_x=0, crop_y=0, num_superpixels=400, compactness=15)
        segments, segmented_image = slic.Slic_segmentation_for_all_pixels(small_image)
        print(f"[STEPP] SLIC produced {len(segments)} unique segments")

        # ---- 3b. Resize segment map to DINO feature resolution (50x50) ----
        resized_seg, new_segment_dict = slic.make_masks_smaller_numpy(
            segments, segmented_image, 50
        )

        # ---- 3c. Average features per segment (GPU-accelerated) ----
        # Convert segment map to torch tensor on GPU
        seg_tensor = torch.from_numpy(resized_seg.astype(np.int64)).to(device)
        average_features = average_dino_feature_segment_tensor(features, seg_tensor)
        print(f"[STEPP] Averaged features shape: {average_features.shape}")

        # ---- 3d. MLP forward pass ----
        with torch.no_grad():
            reconstructed = model(average_features)

        # ---- 3e. Reconstruction error per segment ----
        losses = loss_fn(average_features, reconstructed)
        losses = losses.mean(dim=1).cpu().numpy()

        # ---- 3f. Map losses back to pixel space ----
        cost_map = np.zeros_like(segmented_image, dtype=np.float32)
        for key, loss_val in zip(new_segment_dict.keys(), losses):
            cost_map[segmented_image == int(key)] = loss_val

        # Normalize & threshold
        cost_map = np.clip(cost_map, 0, 10)
        cost_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min() + 1e-8) * 0.45
        cost_map = np.clip(cost_map, 0, threshold)

    elif mode == "pixel_wise":
        # ---- 3a. Direct per-pixel inference (no SLIC) ----
        # features shape: (1, 384, 50, 50)
        features_flat = features.squeeze(0).permute(1, 2, 0).reshape(-1, 384)  # (2500, 384)

        with torch.no_grad():
            reconstructed = model(features_flat)

        # ---- 3b. Reconstruction error per pixel ----
        losses = loss_fn(features_flat, reconstructed)
        losses = losses.mean(dim=1).cpu().numpy()  # (2500,)

        # ---- 3c. Reshape & resize to image size ----
        cost_map = losses.reshape(50, 50)
        cost_map = cv2.resize(cost_map, (H, H))

        # Normalize & threshold
        cost_map = np.clip(cost_map, 0, 10)
        cost_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min() + 1e-8) * 0.45
        cost_map = np.clip(cost_map, 0, threshold)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'segment_wise' or 'pixel_wise'.")

    t_elapsed = time.time() - t_start
    print(f"[STEPP] Inference completed in {t_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 4. Visualization
    # ------------------------------------------------------------------
    # Custom colormap (RdYlBu stretched)
    s = 0.3
    cmap_obj = cm.get_cmap("RdYlBu", 256)
    cmap_arr = np.vstack([
        cmap_obj(np.linspace(0, s, 128)),
        cmap_obj(np.linspace(1 - s, 1.0, 128)),
    ])
    cmap_arr = (cmap_arr[:, :3] * 255).astype(np.uint8)
    cmap_arr = cmap_arr[::-1]  # Reverse: blue = traversable, red = obstacle

    cost_normalized = ((cost_map - cost_map.min())
                       / (cost_map.max() - cost_map.min() + 1e-8) * 255).astype(np.uint8)
    color_mapped = cmap_arr[cost_normalized]

    # Alpha composite
    img_rgba = PILImage.fromarray(small_image).convert("RGBA")
    seg_rgba = PILImage.fromarray(color_mapped).convert("RGBA")
    seg_np = np.array(seg_rgba)
    alpha = (seg_np[:, :, 3] * 0.5).astype(np.uint8)  # 50% transparency
    seg_np[:, :, 3] = alpha
    seg_rgba = PILImage.fromarray(seg_np)

    composited = PILImage.alpha_composite(img_rgba, seg_rgba)
    result_img = composited.convert("RGB").resize((W, H))

    # ------------------------------------------------------------------
    # 5. Two-panel figure: original | result
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(img_rgb)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis("off")

    ax2.imshow(result_img)
    ax2.set_title(
        f"STEPP Traversability ({mode})\n"
        f"Model: {os.path.basename(model_path)[:40]}...\n"
        f"Threshold: {threshold}",
        fontsize=12,
    )
    ax2.axis("off")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Save composite figure
    fig_path = os.path.join(output_dir, f"{base_name}_{mode}_thresh{threshold}_{stamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[STEPP] Figure saved: {fig_path}")

    # Save raw cost map as numpy
    npy_path = os.path.join(output_dir, f"{base_name}_{mode}_costmap_{stamp}.npy")
    np.save(npy_path, cost_map)
    print(f"[STEPP] Cost map saved: {npy_path}")

    # Save composited result only
    result_only_path = os.path.join(output_dir, f"{base_name}_{mode}_overlay_{stamp}.png")
    result_img.save(result_only_path)
    print(f"[STEPP] Overlay image saved: {result_only_path}")

    return fig, result_only_path


# ======================================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Choose checkpoint:
    #   - checkpoints/all_ViT_small_input_700_big_nn_checkpoint_20240827-1935.pth
    #   - checkpoints/richmond_forest_full_ViT_small_big_nn_checkpoint_20240821-1825.pth
    #   - checkpoints/unreal_full_ViT_small_big_nn_checkpoint_20240819-2003.pth
    MODEL_PATH = os.path.join(_PROJ_ROOT, "checkpoints",
                              "all_ViT_small_input_700_big_nn_checkpoint_20240827-1935.pth")

    # Test image — place your images in test_images/ folder
    # The script will use the first image found, or specify a path directly:
    # IMAGE_PATH = "/path/to/your/image.jpg"
    _test_dir = os.path.join(_PROJ_ROOT, "test_images")
    _images = sorted([
        f for f in os.listdir(_test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]) if os.path.isdir(_test_dir) else []
    if not _images:
        print(f"[ERROR] No images found in {_test_dir}/. Please place test images there.")
        sys.exit(1)
    IMAGE_PATH = os.path.join(_test_dir, _images[0])
    print(f"[STEPP] Using first test image: {_images[0]}")

    # Inference mode: "segment_wise" or "pixel_wise"
    MODE = "segment_wise"

    # Traversability threshold (0.0–1.0). Lower = more conservative (more red).
    THRESHOLD = 0.15

    # --- RUN ---
    fig, result_path = run_inference(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        mode=MODE,
        threshold=THRESHOLD,
    )

    plt.close("all")
    print(f"\n[DONE] Result: {result_path}")
