import os
import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from pydicom.uid import generate_uid

# Optional GPU libs
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False

HAS_CV2_CUDA = hasattr(cv2, 'cuda')

# -------------------------
# Utilities: I/O & helpers
# -------------------------
def read_dicom_frames(path):
    """Return frames as numpy float32 array with shape (n_frames, H, W)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # convert to float32
    return arr.astype(np.float32), ds


def read_image_as_gray(path):
    """Read a single image (png/jpg/...) and return float32 grayscale."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)


def normalize_to_float32(img):
    """Normalize an image to range [0,1] as float32. Works per-image."""
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx - mn < 1e-9:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - mn) / (mx - mn)
    return out.astype(np.float32)


# -------------------------
# Contrast Enhancement Methods
# -------------------------
def apply_gamma_correction(img, gamma=1.0):
    """
    Apply gamma correction to enhance contrast.
    """
    if gamma <= 0:
        return img
    img_norm = normalize_to_float32(img)
    corrected = np.power(img_norm, gamma)
    return corrected.astype(np.float32)


def apply_histogram_equalization(img):
    """
    Apply histogram equalization to enhance contrast.
    Works on normalized float32 images.
    """
    img_norm = normalize_to_float32(img)
    img_8bit = (img_norm * 255).astype(np.uint8)
    equalized_8bit = cv2.equalizeHist(img_8bit)
    equalized = equalized_8bit.astype(np.float32) / 255.0
    return equalized


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE on CPU.
    """
    img_norm = normalize_to_float32(img)
    img_8bit = (img_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_8bit = clahe.apply(img_8bit)
    enhanced = enhanced_8bit.astype(np.float32) / 255.0
    return enhanced


def apply_gamma_clahe(img, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    gamma_corrected = apply_gamma_correction(img, gamma)
    enhanced = apply_clahe(gamma_corrected, clip_limit, tile_grid_size)
    return enhanced


def enhance_contrast(img, method='gamma_clahe', gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    if method == 'none':
        return normalize_to_float32(img)
    elif method == 'gamma':
        return apply_gamma_correction(img, gamma)
    elif method == 'hist_eq':
        return apply_histogram_equalization(img)
    elif method == 'clahe':
        return apply_clahe(img, clip_limit, tile_grid_size)
    elif method == 'gamma_clahe':
        return apply_gamma_clahe(img, gamma, clip_limit, tile_grid_size)
    else:
        print(f"Unknown enhancement method: {method}. Using 'gamma_clahe'.")
        return apply_gamma_clahe(img, gamma, clip_limit, tile_grid_size)


def prepare_for_ecc(img, enhancement_method='gamma_clahe', gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """Ensure img is single-channel, float32, normalized, and contrast-enhanced with gamma+CLAHE."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(img, enhancement_method, gamma, clip_limit, tile_grid_size)
    return enhanced.astype(np.float32)


# -------------------------
# Warping helpers
# -------------------------
def apply_warp(img, warp, motion, dsize):
    """Apply warp to img (float32) given motion type. dsize = (w,h)."""
    if motion == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        return cv2.warpAffine(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


# -------------------------
# Decompose warp matrix
# -------------------------
def decompose_warp2d(warp, motion):
    if motion == cv2.MOTION_TRANSLATION:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation': (tx, ty)}
    elif motion == cv2.MOTION_EUCLIDEAN:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        angle_rad = np.arctan2(warp[1,0], warp[0,0])
        return {'translation': (tx, ty), 'rotation_deg': float(np.degrees(angle_rad))}
    elif motion == cv2.MOTION_AFFINE:
        A = warp[:,:2].astype(np.float64)
        tx, ty = float(warp[0,2]), float(warp[1,2])
        sx = np.linalg.norm(A[:,0])
        sy = np.linalg.norm(A[:,1])
        theta = np.arctan2(A[1,0], A[0,0])
        shear = np.dot(A[:,0], A[:,1])/(sx*sy) if sx*sy > 1e-9 else 0.0
        return {'translation': (tx, ty), 'rotation_deg': float(np.degrees(theta)),
                'scale_x': float(sx), 'scale_y': float(sy), 'shear': float(shear)}
    elif motion == cv2.MOTION_HOMOGRAPHY:
        return {'homography': warp.copy()}
    else:
        return {}


# -------------------------
# ECC estimation with gamma+CLAHE preprocessing on both mask and live frames
# -------------------------
def estimate_motion_ecc(mask_frame, live_frame, motion=cv2.MOTION_AFFINE,
                        number_of_iterations=500, termination_eps=1e-6,
                        gaussFiltSize=5, pyr_levels=0, init_warp=None, mask=None,
                        gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Estimate warp that maps `live_frame` -> `mask_frame` using ECC.
    Both mask and live frames are preprocessed with gamma+CLAHE enhancement.
    If pyr_levels > 0, run from coarse to fine (pyramid).
    Returns (ecc_value, warp_matrix)
    """
    print(f"Applying Gamma+CLAHE preprocessing (gamma={gamma}, clip_limit={clip_limit})...")
    mask_enhanced = prepare_for_ecc(mask_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    live_enhanced = prepare_for_ecc(live_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    print(f"Mask frame shape after preprocessing: {mask_enhanced.shape}")
    print(f"Live frame shape after preprocessing: {live_enhanced.shape}")
    h, w = mask_enhanced.shape[:2]

    if motion == cv2.MOTION_HOMOGRAPHY:
        if init_warp is None:
            warp = np.eye(3, dtype=np.float32)
        else:
            warp = init_warp.astype(np.float32)
    else:
        if init_warp is None:
            warp = np.eye(2, 3, dtype=np.float32)
        else:
            warp = init_warp.astype(np.float32)

    levels = list(range(pyr_levels, -1, -1))
    current_warp = warp.copy()
    print(f"Running ECC with {len(levels)} pyramid levels...")

    for level_idx, level in enumerate(levels):
        scale = 1.0 / (2 ** level)
        mask_level = cv2.resize(mask_enhanced, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        live_level = cv2.resize(live_enhanced, (mask_level.shape[1], mask_level.shape[0]),
                               interpolation=cv2.INTER_AREA)
        mask_w = None
        if mask is not None:
            mask_w = cv2.resize(mask.astype(np.uint8), (mask_level.shape[1], mask_level.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        if motion == cv2.MOTION_HOMOGRAPHY:
            if current_warp.shape != (3,3):
                tmp = np.eye(3, dtype=np.float32)
                tmp[:2,:] = current_warp
                current_warp = tmp
        else:
            if current_warp.shape != (2,3):
                tmp = np.eye(2,3, dtype=np.float32)
                tmp[:2,:] = current_warp[:2,:]
                current_warp = tmp
        if level != levels[0]:
            if motion == cv2.MOTION_HOMOGRAPHY:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0
            else:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        try:
            print(f"  Level {level_idx+1}/{len(levels)}: Scale {scale:.2f}, Size {mask_level.shape}")
            res = cv2.findTransformECC(mask_level, live_level, current_warp, motion, criteria, mask_w, gaussFiltSize)
            if isinstance(res, tuple) or isinstance(res, list):
                ecc_val, current_warp = res[0], res[1]
            else:
                ecc_val = float(res)
            print(f"  Level {level_idx+1} ECC: {ecc_val:.4f}")
        except cv2.error as e:
            raise RuntimeError(f"cv2.findTransformECC failed at level {level}: " + str(e))

    print(f"Final ECC value: {ecc_val:.4f}")
    return float(ecc_val), current_warp


# -------------------------
# DSA (Digital Subtraction Angiography) Methods
# -------------------------
def compute_dsa(mask_enhanced, live_aligned, method='linear', weight=1.0):
    if method == 'linear':
        dsa = mask_enhanced - live_aligned
    elif method == 'logarithmic':
        eps = 1e-6
        mask_log = np.log(mask_enhanced + eps)
        live_log = np.log(live_aligned + eps)
        dsa = mask_log - live_log
    elif method == 'weighted':
        dsa = mask_enhanced - (weight * live_aligned)
    else:
        print(f"Unknown DSA method: {method}. Using 'linear'.")
        dsa = mask_enhanced - live_aligned
    return dsa.astype(np.float32)


def enhance_dsa_result(dsa, enhancement_method='clahe', gamma=0.6, clip_limit=3.0,
                      denoise=True, gaussian_sigma=0.8):
    dsa_enhanced = dsa.copy()
    dsa_norm = normalize_to_float32(dsa_enhanced)
    if denoise:
        dsa_norm = ndimage.gaussian_filter(dsa_norm, sigma=gaussian_sigma)
    if enhancement_method == 'clahe':
        dsa_enhanced = apply_clahe(dsa_norm, clip_limit=clip_limit, tile_grid_size=(8,8))
    elif enhancement_method == 'gamma':
        dsa_enhanced = apply_gamma_correction(dsa_norm, gamma=gamma)
    elif enhancement_method == 'hist_eq':
        dsa_enhanced = apply_histogram_equalization(dsa_norm)
    elif enhancement_method == 'gamma_clahe':
        dsa_enhanced = apply_gamma_clahe(dsa_norm, gamma=gamma, clip_limit=clip_limit)
    else:
        dsa_enhanced = dsa_norm
    return dsa_enhanced


def calculate_dsa_quality_metrics(dsa, vessel_roi=None, background_roi=None):
    metrics = {}
    metrics['mean'] = float(np.mean(dsa))
    metrics['std'] = float(np.std(dsa))
    metrics['min'] = float(np.min(dsa))
    metrics['max'] = float(np.max(dsa))
    metrics['dynamic_range'] = metrics['max'] - metrics['min']
    signal = np.abs(np.mean(dsa))
    noise = np.std(dsa)
    if noise > 1e-6:
        metrics['snr'] = signal / noise
    else:
        metrics['snr'] = float('inf')
    if vessel_roi is not None and background_roi is not None:
        vessel_mean = np.mean(dsa[vessel_roi > 0])
        background_mean = np.mean(dsa[background_roi > 0])
        background_std = np.std(dsa[background_roi > 0])
        if background_std > 1e-6:
            metrics['cnr'] = abs(vessel_mean - background_mean) / background_std
        else:
            metrics['cnr'] = float('inf')
    return metrics


# -------------------------
# Enhanced Visualization with DSA
# -------------------------
def visualize_dsa_results(mask_frame, live_frame, warp, motion, dsa_method='linear',
                         gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8),
                         enhance_dsa=True, dsa_enhancement='gamma_clahe'):
    print("Preparing DSA visualization...")
    mask_enhanced = prepare_for_ecc(mask_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    live_enhanced = prepare_for_ecc(live_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    h, w = mask_enhanced.shape[:2]
    live_aligned = apply_warp(live_enhanced, warp, motion, (w, h))
    print(f"Computing DSA using {dsa_method} subtraction...")
    dsa_raw = compute_dsa(mask_enhanced, live_aligned, method=dsa_method)
    if enhance_dsa:
        dsa_enhanced = enhance_dsa_result(dsa_raw, enhancement_method=dsa_enhancement,
                                        gamma=0.6, clip_limit=3.0, denoise=True)
    else:
        dsa_enhanced = normalize_to_float32(dsa_raw)
    dsa_metrics = calculate_dsa_quality_metrics(dsa_raw)
    mask_original = normalize_to_float32(mask_frame)
    live_original = normalize_to_float32(live_frame)
    live_aligned_original = apply_warp(live_original, warp, motion, (w, h))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes[0,0].imshow(mask_original, cmap='gray')
    axes[0,0].set_title('Original Mask Frame')
    axes[0,0].axis('off')
    axes[0,1].imshow(live_original, cmap='gray')
    axes[0,1].set_title('Original Live Frame')
    axes[0,1].axis('off')
    axes[0,2].imshow(live_aligned_original, cmap='gray')
    axes[0,2].set_title('Original Aligned')
    axes[0,2].axis('off')
    axes[0,3].imshow(mask_original - live_aligned_original, cmap='seismic', vmin=-0.5, vmax=0.5)
    axes[0,3].set_title('Original DSA (Linear)')
    axes[0,3].axis('off')
    axes[1,0].imshow(mask_enhanced, cmap='gray')
    axes[1,0].set_title(f'Enhanced Mask (γ={gamma})')
    axes[1,0].axis('off')
    axes[1,1].imshow(live_enhanced, cmap='gray')
    axes[1,1].set_title(f'Enhanced Live (γ={gamma})')
    axes[1,1].axis('off')
    axes[1,2].imshow(live_aligned, cmap='gray')
    axes[1,2].set_title('Enhanced Aligned')
    axes[1,2].axis('off')
    axes[1,3].imshow(dsa_raw, cmap='seismic', vmin=np.percentile(dsa_raw, 5), vmax=np.percentile(dsa_raw, 95))
    axes[1,3].set_title(f'Raw DSA ({dsa_method})')
    axes[1,3].axis('off')
    axes[2,0].imshow(dsa_enhanced, cmap='gray')
    axes[2,0].set_title(f'Enhanced DSA ({dsa_enhancement})')
    axes[2,0].axis('off')
    dsa_inverted = 1.0 - dsa_enhanced
    axes[2,1].imshow(dsa_inverted, cmap='gray')
    axes[2,1].set_title('Inverted DSA (Bright Vessels)')
    axes[2,1].axis('off')
    axes[2,2].imshow(dsa_enhanced, cmap='hot')
    axes[2,2].set_title('Color-mapped DSA')
    axes[2,2].axis('off')
    dsa_abs = np.abs(dsa_raw)
    axes[2,3].imshow(dsa_abs, cmap='gray')
    axes[2,3].set_title('Absolute DSA')
    axes[2,3].axis('off')
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im1 = axes[0,0].imshow(dsa_enhanced, cmap='gray')
    axes[0,0].set_title(f'Enhanced DSA ({dsa_enhancement})')
    plt.colorbar(im1, ax=axes[0,0])
    im2 = axes[0,1].imshow(dsa_inverted, cmap='gray')
    axes[0,1].set_title('Inverted DSA (Vessels Bright)')
    plt.colorbar(im2, ax=axes[0,1])
    im3 = axes[1,0].imshow(dsa_enhanced, cmap='jet')
    axes[1,0].set_title('Jet Colormap DSA')
    plt.colorbar(im3, ax=axes[1,0])
    axes[1,1].hist(dsa_raw.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[1,1].set_title('DSA Value Distribution')
    axes[1,1].set_xlabel('DSA Intensity')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"\nDSA Quality Metrics:")
    print(f"Method: {dsa_method} subtraction")
    print(f"Enhancement: {dsa_enhancement if enhance_dsa else 'None'}")
    print("-" * 40)
    for key, value in dsa_metrics.items():
        if isinstance(value, float):
            print(f"{key.upper()}: {value:.4f}")
        else:
            print(f"{key.upper()}: {value}")
    return dsa_raw, dsa_enhanced


def compare_dsa_methods(mask_enhanced, live_aligned, methods=['linear', 'logarithmic', 'weighted']):
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    for i, method in enumerate(methods):
        if method == 'weighted':
            dsa = compute_dsa(mask_enhanced, live_aligned, method=method, weight=0.9)
        else:
            dsa = compute_dsa(mask_enhanced, live_aligned, method=method)
        dsa_enhanced = enhance_dsa_result(dsa, enhancement_method='gamma_clahe')
        axes[0,i].imshow(dsa, cmap='seismic', vmin=np.percentile(dsa, 5), vmax=np.percentile(dsa, 95))
        axes[0,i].set_title(f'Raw DSA ({method.title()})')
        axes[0,i].axis('off')
        axes[1,i].imshow(dsa_enhanced, cmap='gray')
        axes[1,i].set_title(f'Enhanced DSA ({method.title()})')
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.show()


def compare_enhancement_methods(img, methods=['none', 'gamma', 'clahe', 'gamma_clahe']):
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    for i, method in enumerate(methods):
        enhanced = enhance_contrast(img, method)
        axes[i].imshow(enhanced, cmap='gray')
        title = f'{method.replace("_", " ").title()}'
        if method == 'gamma_clahe':
            title += ' (Recommended)'
            axes[i].set_title(title, fontweight='bold', color='red')
        else:
            axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# -------------------------
# GPU helpers
# -------------------------
def gpu_normalize(img_gpu):
    """Normalize single image on GPU (cupy array) to [0,1]"""
    if not HAS_CUPY:
        raise RuntimeError('cupy not available')
    mn = img_gpu.min()
    mx = img_gpu.max()
    denom = mx - mn
    if float(denom) < 1e-9:
        return cp.zeros_like(img_gpu, dtype=cp.float32)
    out = (img_gpu - mn) / denom
    return out.astype(cp.float32)


def gpu_gamma(img_gpu, gamma=1.0):
    if not HAS_CUPY:
        raise RuntimeError('cupy not available')
    if gamma <= 0:
        return img_gpu
    out = cp.power(img_gpu, gamma)
    return out


def gpu_clahe_batch(frames_gpu, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE on GPU using cv2.cuda if available. frames_gpu is a list or array of GPU numpy (cupy) arrays
    If cv2.cuda.createCLAHE not available, fallback to CPU CLAHE per-frame.
    Returns list of numpy float32 frames (on CPU) in range [0,1].
    """
    results = []
    if HAS_CV2_CUDA and hasattr(cv2.cuda, 'createCLAHE'):
        try:
            clahe_gpu = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            for i in range(len(frames_gpu)):
                # move to CPU uint8 (cv2.cuda CLAHE expects GpuMat U8)
                frame_cpu = cp.asnumpy((frames_gpu[i] * 255.0).astype(cp.uint8))
                gmat = cv2.cuda_GpuMat()
                gmat.upload(frame_cpu)
                out_gpu = clahe_gpu.apply(gmat)
                out = out_gpu.download().astype(np.float32) / 255.0
                results.append(out)
            return results
        except Exception as e:
            print('cv2.cuda CLAHE failed or not usable, falling back to CPU CLAHE:', e)

    # Fallback: CPU CLAHE using apply_clahe
    for i in range(len(frames_gpu)):
        frame_cpu = cp.asnumpy(frames_gpu[i]) if HAS_CUPY else frames_gpu[i]
        # input expected in [0,1]
        results.append(apply_clahe(frame_cpu, clip_limit=clip_limit, tile_grid_size=tile_grid_size))
    return results


def gpu_preprocess_frames(frames, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Preprocess a stack of frames on GPU if available:
    Steps: convert to GPU arrays -> normalize -> gamma -> CLAHE (if available) -> return CPU float32 frames [0,1]
    """
    n, h, w = frames.shape
    print(f"Preprocessing {n} frames on GPU: CUPY={HAS_CUPY}, CV2_CUDA={HAS_CV2_CUDA}")
    frames_gpu = []
    for i in range(n):
        f = frames[i].astype(np.float32)
        if HAS_CUPY:
            g = cp.asarray(f)
            g = gpu_normalize(g)
            g = gpu_gamma(g, gamma)
            frames_gpu.append(g)
        else:
            # CPU fallback: normalize and gamma
            tmp = normalize_to_float32(f)
            tmp = np.power(tmp, gamma)
            frames_gpu.append(tmp)
    # Apply CLAHE per-frame (GPU where possible)
    processed = gpu_clahe_batch(frames_gpu, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    # Ensure float32
    processed_np = np.stack([p.astype(np.float32) for p in processed], axis=0)
    return processed_np


# -------------------------
# Multi-frame DSA pipeline
# -------------------------
def process_all_frames_and_save_cine(dicom_path, output_folder='output', mask_frame_idx=0,
                                     motion_model=cv2.MOTION_AFFINE, pyr_levels=0,
                                     gamma_value=0.8, clahe_clip_limit=2.0, clahe_tile_size=(8,8),
                                     dsa_method='linear', enable_dsa_enhancement=True, dsa_enhancement_method='gamma_clahe'):
    os.makedirs(output_folder, exist_ok=True)
    frames, ds = read_dicom_frames(dicom_path)
    n_frames, H, W = frames.shape
    print(f"Loaded DICOM: {dicom_path} -> {n_frames} frames, size=({H},{W})")

    # GPU preprocess all frames (normalize, gamma, CLAHE) -> returns CPU float32 frames in [0,1]
    preprocessed = gpu_preprocess_frames(frames, gamma=gamma_value, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_size)

    # Prepare mask (reference) frame
    mask_frame = preprocessed[mask_frame_idx]

    # We'll compute ECC for each frame (live) against mask_frame. ECC is CPU-only.
    # Pre-allocate arrays to store aligned frames and DSA frames
    aligned_frames = np.zeros_like(preprocessed)
    dsa_frames = np.zeros_like(preprocessed)

    # We can try to reuse init_warp from previous frame to accelerate (optional)
    init_warp = None

    for i in range(n_frames):
        if i == mask_frame_idx:
            aligned_frames[i] = preprocessed[i]
            dsa_frames[i] = np.zeros_like(preprocessed[i])  # or blank for mask
            print(f"Frame {i}: (mask) skipped ECC.")
            continue

        live_frame = preprocessed[i]
        print(f"\nEstimating motion for frame {i}/{n_frames-1} -> ECC on CPU...")
        try:
            ecc_val, warp = estimate_motion_ecc(mask_frame, live_frame, motion=motion_model,
                                               number_of_iterations=200, termination_eps=1e-6,
                                               gaussFiltSize=5, pyr_levels=pyr_levels,
                                               init_warp=init_warp,
                                               gamma=gamma_value, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_size)
            init_warp = warp  # reuse as starting guess
            # Apply warp to original (enhanced) live frame to obtain aligned
            h, w = mask_frame.shape[:2]
            live_aligned = apply_warp(live_frame, warp, motion_model, (w, h))
            aligned_frames[i] = live_aligned
            # Compute DSA (mask - aligned) on CPU; could be done on GPU but keep simple
            dsa_raw = compute_dsa(mask_frame, live_aligned, method=dsa_method)
            if enable_dsa_enhancement:
                dsa_enh = enhance_dsa_result(dsa_raw, enhancement_method=dsa_enhancement_method)
            else:
                dsa_enh = normalize_to_float32(dsa_raw)
            dsa_frames[i] = dsa_enh
            print(f"Frame {i}: ECC={ecc_val:.6f}")
        except Exception as e:
            print(f"Frame {i}: ECC failed: {e}. Using unaligned frame for DSA.")
            aligned_frames[i] = live_frame
            dsa_frames[i] = normalize_to_float32(mask_frame - live_frame)

    # Build cine DSA: exclude mask frame or keep it as zeros — here we include all frames but mask has zeros
    # Convert to uint16 for saving in DICOM
    # Normalize each DSA frame to full uint16 range
    print("Scaling DSA frames to uint16 for DICOM storage...")
    cine = np.zeros_like(dsa_frames, dtype=np.uint16)
    for i in range(n_frames):
        arr = dsa_frames[i]
        arr_n = normalize_to_float32(arr)
        cine[i] = (arr_n * 65535.0).astype(np.uint16)

    # Prepare new DICOM dataset for output
    out_ds = ds
    # Update required tags for multi-frame pixel data
    out_ds.NumberOfFrames = str(n_frames)
    out_ds.Rows = int(H)
    out_ds.Columns = int(W)
    out_ds.SamplesPerPixel = 1
    out_ds.PhotometricInterpretation = 'MONOCHROME2'
    out_ds.BitsAllocated = 16
    out_ds.BitsStored = 16
    out_ds.HighBit = 15
    out_ds.PixelRepresentation = 0  # unsigned
    # UIDs
    out_ds.SOPInstanceUID = generate_uid()
    out_ds.SeriesInstanceUID = generate_uid()
    out_ds.StudyInstanceUID = getattr(out_ds, 'StudyInstanceUID', generate_uid())

    # PixelData
    out_ds.PixelData = cine.tobytes()

    # Save
    out_path = os.path.join(output_folder, 'cine_dsa.dcm')
    pydicom.dcmwrite(out_path, out_ds)
    print(f"Saved cine DSA DICOM to: {out_path}")

    return out_path, aligned_frames, dsa_frames


# -------------------------
# Example usage (keeps same structure as original script)
# -------------------------
if __name__ == "__main__":
    # Modify the DICOM path to your own
    dicom_path = "D:/Rohith/RAW_AUTOPIXEL/Neuro_RAW/1.2.826.0.1.3680043.2.1330.2640165.2408301133290002.5.445_Raw_anon.dcm"

    # Choose mask frame (reference) and settings
    mask_frame_idx = 0
    motion_model = cv2.MOTION_AFFINE
    pyr_levels = 1
    gamma_value = 0.8
    clahe_clip_limit = 2.0
    clahe_tile_size = (8, 8)
    dsa_method = 'linear'
    enable_dsa_enhancement = True
    dsa_enhancement_method = 'gamma_clahe'

    print("="*70)
    print("ECC Motion Estimation + DSA with GPU preprocessing (when available)")
    print("="*70)

    out_path, aligned, dsa = process_all_frames_and_save_cine(
        dicom_path,
        output_folder='output',
        mask_frame_idx=mask_frame_idx,
        motion_model=motion_model,
        pyr_levels=pyr_levels,
        gamma_value=gamma_value,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        dsa_method=dsa_method,
        enable_dsa_enhancement=enable_dsa_enhancement,
        dsa_enhancement_method=dsa_enhancement_method
    )

    print("Processing complete.")
