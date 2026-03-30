"""
Estimate motion between two frames (DICOM or image files) using OpenCV ECC.
Enhanced with gamma+CLAHE preprocessing and DSA (Digital Subtraction Angiography).
Requirements:
    pip install opencv-python pydicom matplotlib numpy scipy
"""

import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage

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
    return arr.astype(np.float32)

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
    gamma < 1.0: brighten image (expand dark regions)
    gamma > 1.0: darken image (compress bright regions)
    gamma = 1.0: no change
    """
    if gamma <= 0:
        return img
    
    # Ensure image is normalized to [0,1]
    img_norm = normalize_to_float32(img)
    
    # Apply gamma correction
    corrected = np.power(img_norm, gamma)
    return corrected.astype(np.float32)

def apply_histogram_equalization(img):
    """
    Apply histogram equalization to enhance contrast.
    Works on normalized float32 images.
    """
    # Convert to 8-bit for OpenCV histogram equalization
    img_norm = normalize_to_float32(img)
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # Apply histogram equalization
    equalized_8bit = cv2.equalizeHist(img_8bit)
    
    # Convert back to float32 [0,1]
    equalized = equalized_8bit.astype(np.float32) / 255.0
    return equalized

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Better than regular histogram equalization for medical images.
    """
    # Convert to 8-bit for OpenCV CLAHE
    img_norm = normalize_to_float32(img)
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced_8bit = clahe.apply(img_8bit)
    
    # Convert back to float32 [0,1]
    enhanced = enhanced_8bit.astype(np.float32) / 255.0
    return enhanced

def apply_gamma_clahe(img, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply gamma correction first, then CLAHE - optimized for medical images.
    This is now the primary enhancement method.
    """
    # Step 1: Apply gamma correction
    gamma_corrected = apply_gamma_correction(img, gamma)
    
    # Step 2: Apply CLAHE to gamma-corrected image
    enhanced = apply_clahe(gamma_corrected, clip_limit, tile_grid_size)
    
    return enhanced

def enhance_contrast(img, method='gamma_clahe', gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply contrast enhancement using specified method.
    
    Parameters:
    - method: 'gamma', 'hist_eq', 'clahe', 'gamma_clahe', or 'none'
    - gamma: gamma value for gamma correction
    - clip_limit: clip limit for CLAHE
    - tile_grid_size: tile grid size for CLAHE
    """
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
    
    # Apply gamma+CLAHE enhancement (recommended for medical images)
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
    """
    Decode/approximate the geometric parameters from the warp matrix.
    Returns a dict with keys depending on motion type.
    """
    if motion == cv2.MOTION_TRANSLATION:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation': (tx, ty)}
    elif motion == cv2.MOTION_EUCLIDEAN:
        # warp is [ cos -sin tx; sin cos ty ]
        tx, ty = float(warp[0,2]), float(warp[1,2])
        angle_rad = np.arctan2(warp[1,0], warp[0,0])
        return {'translation': (tx, ty), 'rotation_deg': float(np.degrees(angle_rad))}
    elif motion == cv2.MOTION_AFFINE:
        A = warp[:,:2].astype(np.float64)  # 2x2
        tx, ty = float(warp[0,2]), float(warp[1,2])
        # scales (approx)
        sx = np.linalg.norm(A[:,0])
        sy = np.linalg.norm(A[:,1])
        # rotation approx from first column
        theta = np.arctan2(A[1,0], A[0,0])
        # shear (cosine between columns)
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
    
    # Apply gamma+CLAHE preprocessing to both frames
    mask_enhanced = prepare_for_ecc(mask_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    live_enhanced = prepare_for_ecc(live_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    
    print(f"Mask frame shape after preprocessing: {mask_enhanced.shape}")
    print(f"Live frame shape after preprocessing: {live_enhanced.shape}")
    
    h, w = mask_enhanced.shape[:2]

    # initial warp
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
        
        # Resize enhanced frames for this pyramid level
        mask_level = cv2.resize(mask_enhanced, (max(1, int(w * scale)), max(1, int(h * scale))), 
                               interpolation=cv2.INTER_AREA)
        live_level = cv2.resize(live_enhanced, (mask_level.shape[1], mask_level.shape[0]), 
                               interpolation=cv2.INTER_AREA)
        
        mask_w = None
        if mask is not None:
            mask_w = cv2.resize(mask.astype(np.uint8), (mask_level.shape[1], mask_level.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)

        # prepare warp for this level:
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

        # Scale translation parameters for pyramid level
        if level != levels[0]:  
            if motion == cv2.MOTION_HOMOGRAPHY:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0
            else:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0

        # run ECC on preprocessed frames
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        try:
            print(f"  Level {level_idx+1}/{len(levels)}: Scale {scale:.2f}, Size {mask_level.shape}")
            
            res = cv2.findTransformECC(mask_level, live_level, current_warp, motion, criteria, mask_w, gaussFiltSize)
            if isinstance(res, tuple) or isinstance(res, list):
                ecc_val, current_warp = res[0], res[1]
            else:
                # single value returned; warp updated in-place
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
    """
    Compute DSA using various subtraction methods.
    
    Parameters:
    - mask_enhanced: Preprocessed mask frame (reference/baseline)
    - live_aligned: Aligned and preprocessed live frame (with contrast)
    - method: 'linear', 'logarithmic', 'weighted'
    - weight: Weight factor for weighted subtraction
    
    Returns:
    - DSA image (float32)
    """
    if method == 'linear':
        # Standard linear subtraction: DSA = Mask - Live
        dsa = mask_enhanced - live_aligned
        
    elif method == 'logarithmic':
        # Logarithmic subtraction: DSA = log(Mask) - log(Live) = log(Mask/Live)
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        mask_log = np.log(mask_enhanced + eps)
        live_log = np.log(live_aligned + eps)
        dsa = mask_log - live_log
        
    elif method == 'weighted':
        # Weighted subtraction: DSA = Mask - (weight * Live)
        dsa = mask_enhanced - (weight * live_aligned)
        
    else:
        print(f"Unknown DSA method: {method}. Using 'linear'.")
        dsa = mask_enhanced - live_aligned
    
    return dsa.astype(np.float32)

def enhance_dsa_result(dsa, enhancement_method='clahe', gamma=0.6, clip_limit=3.0, 
                      denoise=True, gaussian_sigma=0.8):
    """
    Enhance DSA result for better vessel visibility.
    
    Parameters:
    - dsa: Raw DSA image
    - enhancement_method: 'clahe', 'gamma', 'hist_eq', or 'none'
    - gamma: Gamma value for vessel enhancement (< 1.0 to brighten vessels)
    - clip_limit: CLAHE clip limit
    - denoise: Apply denoising
    - gaussian_sigma: Gaussian filter sigma for denoising
    
    Returns:
    - Enhanced DSA image
    """
    dsa_enhanced = dsa.copy()
    
    # Step 1: Normalize DSA to [0,1] range
    dsa_norm = normalize_to_float32(dsa_enhanced)
    
    # Step 2: Apply denoising if requested
    if denoise:
        dsa_norm = ndimage.gaussian_filter(dsa_norm, sigma=gaussian_sigma)
    
    # Step 3: Apply contrast enhancement
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
    """
    Calculate quality metrics for DSA result.
    
    Parameters:
    - dsa: DSA image
    - vessel_roi: ROI mask for vessel regions (optional)
    - background_roi: ROI mask for background regions (optional)
    
    Returns:
    - Dictionary with quality metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = float(np.mean(dsa))
    metrics['std'] = float(np.std(dsa))
    metrics['min'] = float(np.min(dsa))
    metrics['max'] = float(np.max(dsa))
    metrics['dynamic_range'] = metrics['max'] - metrics['min']
    
    # Signal-to-Noise Ratio (rough estimate)
    signal = np.abs(np.mean(dsa))
    noise = np.std(dsa)
    if noise > 1e-6:
        metrics['snr'] = signal / noise
    else:
        metrics['snr'] = float('inf')
    
    # If ROIs are provided, calculate CNR
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
    """
    Comprehensive visualization of DSA results.
    Shows: Original frames, Enhanced frames, Aligned result, DSA, and Enhanced DSA.
    """
    print("Preparing DSA visualization...")
    
    # Apply gamma+CLAHE preprocessing to both frames
    mask_enhanced = prepare_for_ecc(mask_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    live_enhanced = prepare_for_ecc(live_frame, 'gamma_clahe', gamma, clip_limit, tile_grid_size)
    
    h, w = mask_enhanced.shape[:2]
    
    # Apply warp to the enhanced live frame
    live_aligned = apply_warp(live_enhanced, warp, motion, (w, h))
    
    # Compute DSA
    print(f"Computing DSA using {dsa_method} subtraction...")
    dsa_raw = compute_dsa(mask_enhanced, live_aligned, method=dsa_method)
    
    # Enhance DSA result
    if enhance_dsa:
        dsa_enhanced = enhance_dsa_result(dsa_raw, enhancement_method=dsa_enhancement, 
                                        gamma=0.6, clip_limit=3.0, denoise=True)
    else:
        dsa_enhanced = normalize_to_float32(dsa_raw)
    
    # Calculate DSA quality metrics
    dsa_metrics = calculate_dsa_quality_metrics(dsa_raw)
    
    # Prepare original normalized versions for comparison
    mask_original = normalize_to_float32(mask_frame)
    live_original = normalize_to_float32(live_frame)
    live_aligned_original = apply_warp(live_original, warp, motion, (w, h))
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Original frames
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
    
    # Row 2: Enhanced frames (Gamma+CLAHE)
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
    
    # Row 3: DSA Analysis
    axes[2,0].imshow(dsa_enhanced, cmap='gray')
    axes[2,0].set_title(f'Enhanced DSA ({dsa_enhancement})')
    axes[2,0].axis('off')
    
    # Inverted DSA (vessels appear bright)
    dsa_inverted = 1.0 - dsa_enhanced
    axes[2,1].imshow(dsa_inverted, cmap='gray')
    axes[2,1].set_title('Inverted DSA (Bright Vessels)')
    axes[2,1].axis('off')
    
    # Color-mapped DSA
    axes[2,2].imshow(dsa_enhanced, cmap='hot')
    axes[2,2].set_title('Color-mapped DSA')
    axes[2,2].axis('off')
    
    # Absolute DSA
    dsa_abs = np.abs(dsa_raw)
    axes[2,3].imshow(dsa_abs, cmap='gray')
    axes[2,3].set_title('Absolute DSA')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Separate detailed DSA plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Enhanced DSA with colorbar
    im1 = axes[0,0].imshow(dsa_enhanced, cmap='gray')
    axes[0,0].set_title(f'Enhanced DSA ({dsa_enhancement})')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Inverted DSA with colorbar  
    im2 = axes[0,1].imshow(dsa_inverted, cmap='gray')
    axes[0,1].set_title('Inverted DSA (Vessels Bright)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Color-mapped with vessels highlighted
    im3 = axes[1,0].imshow(dsa_enhanced, cmap='jet')
    axes[1,0].set_title('Jet Colormap DSA')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Histogram of DSA values
    axes[1,1].hist(dsa_raw.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[1,1].set_title('DSA Value Distribution')
    axes[1,1].set_xlabel('DSA Intensity')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print DSA statistics and quality metrics
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
    """Compare different DSA subtraction methods side by side."""
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for i, method in enumerate(methods):
        # Compute DSA
        if method == 'weighted':
            dsa = compute_dsa(mask_enhanced, live_aligned, method=method, weight=0.9)
        else:
            dsa = compute_dsa(mask_enhanced, live_aligned, method=method)
        
        # Enhanced version - use the function name directly
        dsa_enhanced = enhance_dsa_result(dsa, enhancement_method='gamma_clahe')
        
        # Raw DSA
        axes[0,i].imshow(dsa, cmap='seismic', vmin=np.percentile(dsa, 5), vmax=np.percentile(dsa, 95))
        axes[0,i].set_title(f'Raw DSA ({method.title()})')
        axes[0,i].axis('off')
        
        # Enhanced DSA
        axes[1,i].imshow(dsa_enhanced, cmap='gray')
        axes[1,i].set_title(f'Enhanced DSA ({method.title()})')
        axes[1,i].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_enhancement_methods(img, methods=['none', 'gamma', 'clahe', 'gamma_clahe']):
    """Compare different enhancement methods side by side, highlighting gamma+CLAHE."""
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
# Example usage
# -------------------------
if __name__ == "__main__":
    dicom_path = "D:/Rohith/RAW_AUTOPIXEL/Neuro_RAW/1.2.826.0.1.3680043.2.1330.2640165.2408301133290002.5.445_Raw_anon.dcm"  
    
    # Choose frames (mask frame and live frame)
    mask_frame_idx = 0   # Reference/template frame (pre-contrast)
    live_frame_idx = 20  # Frame to align to mask (post-contrast)
    
    # Motion model
    motion_model = cv2.MOTION_AFFINE  # TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY
    pyr_levels = 1  # 0 = single scale; 1-2 helpful for larger motions
    
    # Gamma+CLAHE Enhancement parameters (optimized for medical images)
    gamma_value = 0.8      # < 1.0 brightens dark regions, > 1.0 darkens
    clahe_clip_limit = 2.0 # Higher values = more contrast enhancement
    clahe_tile_size = (8, 8) # Smaller tiles = more local adaptation
    
    # DSA parameters
    dsa_method = 'linear'  # 'linear', 'logarithmic', 'weighted'
    enable_dsa_enhancement = True  # Changed variable name to avoid conflict
    dsa_enhancement_method = 'gamma_clahe'  # Enhancement method for DSA result

    print("="*70)
    print("ECC Motion Estimation + DSA with Gamma+CLAHE Preprocessing")
    print("="*70)
    
    # Load frames
    frames = read_dicom_frames(dicom_path)  # shape (n, H, W)
    print(f"Loaded frames: {frames.shape}")
    
    mask_frame = frames[mask_frame_idx]
    live_frame = frames[live_frame_idx]
    
    print(f"Processing: Mask frame {mask_frame_idx} vs Live frame {live_frame_idx}")
    print(f"Frame dimensions: {mask_frame.shape}")

    # Optional: Compare different enhancement methods
    print("\nComparing enhancement methods on mask frame...")
    compare_enhancement_methods(mask_frame)

    # Estimate motion with gamma+CLAHE preprocessing on both frames
    print(f"\nRunning ECC with Gamma+CLAHE preprocessing...")
    print(f"Parameters: gamma={gamma_value}, CLAHE clip_limit={clahe_clip_limit}, tile_size={clahe_tile_size}")
    
    ecc, warp = estimate_motion_ecc(
        mask_frame, live_frame, 
        motion=motion_model, 
        pyr_levels=pyr_levels,
        number_of_iterations=200, 
        termination_eps=1e-6, 
        gaussFiltSize=5,
        gamma=gamma_value, 
        clip_limit=clahe_clip_limit, 
        tile_grid_size=clahe_tile_size
    )

    print(f"\nECC Registration Results:")
    print(f"ECC (correlation coefficient): {ecc:.6f}")
    print(f"Estimated warp matrix:")
    print(warp)

    # Decompose transformation parameters
    params = decompose_warp2d(warp, motion_model)
    print(f"\nDecomposed parameters:")
    for key, value in params.items():
        if isinstance(value, tuple):
            print(f"  {key}: ({value[0]:.4f}, {value[1]:.4f})")
        else:
            print(f"  {key}: {value:.4f}")

    # Apply preprocessing to both frames for DSA
    print(f"\nPreparing frames for DSA...")
    mask_enhanced = prepare_for_ecc(mask_frame, 'gamma_clahe', gamma_value, clahe_clip_limit, clahe_tile_size)
    live_enhanced = prepare_for_ecc(live_frame, 'gamma_clahe', gamma_value, clahe_clip_limit, clahe_tile_size)
    
    # Apply alignment to live frame
    h, w = mask_enhanced.shape[:2]
    live_aligned = apply_warp(live_enhanced, warp, motion_model, (w, h))
    
    # Optional: Compare different DSA methods
    print(f"\nComparing DSA subtraction methods...")
    compare_dsa_methods(mask_enhanced, live_aligned)

    # Visualize complete DSA results
    print(f"\nVisualizing DSA results...")
    print(f"DSA method: {dsa_method}")
    print(f"DSA enhancement: {dsa_enhancement_method if enable_dsa_enhancement else 'None'}")
    
    dsa_raw, dsa_enhanced = visualize_dsa_results(
        mask_frame, live_frame, warp, motion_model, 
        dsa_method=dsa_method,
        gamma=gamma_value, 
        clip_limit=clahe_clip_limit, 
        tile_grid_size=clahe_tile_size,
        enhance_dsa=enable_dsa_enhancement,
        dsa_enhancement=dsa_enhancement_method
    )
    
    print(f"\nDSA Processing Complete!")
    print(f"Raw DSA shape: {dsa_raw.shape}")
    print(f"Enhanced DSA shape: {dsa_enhanced.shape}")
    print("="*70)
    