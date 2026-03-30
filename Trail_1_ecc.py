"""
Estimate motion between two frames (DICOM or image files) using OpenCV ECC.
Requirements:
    pip install opencv-python pydicom matplotlib numpy
"""

import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt

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

def prepare_for_ecc(img):
    """Ensure img is single-channel, float32, normalized."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return normalize_to_float32(img).astype(np.float32)

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
# ECC estimation (optionally multi-scale)
# -------------------------
def estimate_motion_ecc(template, frame, motion=cv2.MOTION_AFFINE,
                        number_of_iterations=500, termination_eps=1e-6,
                        gaussFiltSize=5, pyr_levels=0, init_warp=None, mask=None):
    """
    Estimate warp that maps `frame` -> `template` using ECC.
    If pyr_levels > 0, run from coarse to fine (pyramid).
    Returns (ecc_value, warp_matrix)
    """
    # prepare images
    T = prepare_for_ecc(template)
    I = prepare_for_ecc(frame)
    h, w = T.shape[:2]

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

    for level in levels:
        scale = 1.0 / (2 ** level)
        Tw = cv2.resize(T, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        Iw = cv2.resize(I, (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_AREA)
        mask_w = None
        if mask is not None:
            mask_w = cv2.resize(mask.astype(np.uint8), (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_NEAREST)

        
        if level != pyr_levels: 
            pass  

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

        
        if level != levels[0]:  
            if motion == cv2.MOTION_HOMOGRAPHY:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0
            else:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0

        # run ECC on Tw (template) and Iw (input)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        try:
            
            res = cv2.findTransformECC(Tw, Iw, current_warp, motion, criteria, mask_w, gaussFiltSize)
            if isinstance(res, tuple) or isinstance(res, list):
                ecc_val, current_warp = res[0], res[1]
            else:
                # single value returned; warp updated in-place
                ecc_val = float(res)
        except cv2.error as e:
            raise RuntimeError("cv2.findTransformECC failed: " + str(e))

    return float(ecc_val), current_warp

# -------------------------
# Visualization utility
# -------------------------
def visualize_alignment(template, frame, warp, motion):
    h, w = template.shape[:2]
    aligned = apply_warp(frame, warp, motion, (w, h))
    diff = (template - aligned)
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(template, cmap='gray'); ax[0].set_title('Template')
    ax[1].imshow(frame, cmap='gray'); ax[1].set_title('Input')
    ax[2].imshow(aligned, cmap='gray'); ax[2].set_title('Aligned')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.title('Template - Aligned (difference)')
    plt.imshow(diff, cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.show()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"  
    # choose two frame indices from the DICOM loop
    frame_idx_1 = 0
    frame_idx_2 = 10
    # choose motion model
    motion_model = cv2.MOTION_AFFINE  # try TRANSLATION -> EUCLIDEAN -> AFFINE
    pyr_levels = 1  # 0 = single scale; 1-2 helpful for larger motions

    # load frames
    frames = read_dicom_frames(dicom_path)  # shape (n, H, W)
    print("Loaded frames:", frames.shape)
    T = frames[frame_idx_1]
    I = frames[frame_idx_2]

    ecc, warp = estimate_motion_ecc(T, I, motion=motion_model, pyr_levels=pyr_levels,
                                    number_of_iterations=200, termination_eps=1e-6, gaussFiltSize=5)

    print("ECC (correlation coefficient):", ecc)
    print("Estimated warp matrix:\n", warp)

    params = decompose_warp2d(warp, motion_model)
    print("Decomposed params:", params)

    visualize_alignment(prepare_for_ecc(T), prepare_for_ecc(I), warp, motion_model)
