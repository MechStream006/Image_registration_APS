import cv2
import numpy as np

def prepare_for_ecc(img, enhancement_method='gamma_clahe', gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8)):
    """Ensure img is single-channel, float32, normalized, and contrast-enhanced with gamma+CLAHE."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

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