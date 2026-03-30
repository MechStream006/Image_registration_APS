"""
Extended Trail_1_ecc: adds phase-correlation, dense optical flow, warping and Digital Subtraction Angiography (DSA)

Usage: edit the dicom_path and frame indices in the __main__ section and run.
Requires: opencv-python, pydicom, matplotlib, numpy
"""

import os
import csv
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
    return arr.astype(np.float32)


def read_image_as_gray(path):
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
# Warping helpers (original + extended)
# -------------------------

def apply_warp(img, warp, motion, dsize):
    """Apply warp to img (float32) given motion type. dsize = (w,h)."""
    if warp is None:
        return img.copy()

    if motion == cv2.MOTION_HOMOGRAPHY or (warp.shape[0] == 3 and warp.shape[1] == 3):
        return cv2.warpPerspective(
            img, warp.astype(np.float32), dsize,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    elif warp.shape[0] == 2 and warp.shape[1] == 3:
        return cv2.warpAffine(
            img, warp.astype(np.float32), dsize,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        raise ValueError(f"Unsupported warp shape: {warp.shape}")
# -------------------------
# Decompose warp matrix
# -------------------------

def decompose_warp2d(warp, motion):
    if warp is None:
        return {}
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
# ECC estimation
# -------------------------

def estimate_motion_ecc(template, frame, motion=cv2.MOTION_AFFINE,
                        number_of_iterations=500, termination_eps=1e-6,
                        gaussFiltSize=5, pyr_levels=0, init_warp=None, mask=None):
    T = prepare_for_ecc(template)
    I = prepare_for_ecc(frame)
    h, w = T.shape[:2]

    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, dtype=np.float32) if init_warp is None else init_warp.astype(np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32) if init_warp is None else init_warp.astype(np.float32)

    levels = list(range(pyr_levels, -1, -1))
    current_warp = warp.copy()

    for level in levels:
        scale = 1.0 / (2 ** level)
        Tw = cv2.resize(T, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        Iw = cv2.resize(I, (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_AREA)
        mask_w = None
        if mask is not None:
            mask_w = cv2.resize(mask.astype(np.uint8), (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_NEAREST)

        if motion == cv2.MOTION_HOMOGRAPHY and current_warp.shape != (3,3):
            tmp = np.eye(3, dtype=np.float32)
            tmp[:2,:] = current_warp
            current_warp = tmp
        elif motion != cv2.MOTION_HOMOGRAPHY and current_warp.shape != (2,3):
            tmp = np.eye(2,3, dtype=np.float32)
            tmp[:2,:] = current_warp[:2,:]
            current_warp = tmp

        if level != levels[0]:
            current_warp[0,2] *= 2.0
            current_warp[1,2] *= 2.0

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        ecc_val, current_warp = cv2.findTransformECC(Tw, Iw, current_warp, motion, criteria, mask_w, gaussFiltSize)

    return float(ecc_val), current_warp

# -------------------------
# Phase correlation, dense flow & DSA
# -------------------------

def estimate_translation_phasecorr(template, frame):
    a = normalize_to_float32(template)
    b = normalize_to_float32(frame)
    window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    a_w = (a * window).astype(np.float32)
    b_w = (b * window).astype(np.float32)
    shift, response = cv2.phaseCorrelate(a_w, b_w)
    dx, dy = shift
    return float(dx), float(dy), float(response)


def compute_optical_flow(template, frame, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
    a = normalize_to_float32(template)
    b = normalize_to_float32(frame)
    flow = cv2.calcOpticalFlowFarneback(b, a, None,
                                        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return flow, mag


def motion_summary_from_flow(flow, mag, threshold=0.5, max_points=500):
    dx_med = float(np.median(flow[...,0]))
    dy_med = float(np.median(flow[...,1]))
    coords = np.argwhere(mag > threshold)
    if coords.size == 0:
        return {'dx_median': dx_med, 'dy_median': dy_med, 'points': [], 'num_moving': 0}
    coords = coords[np.argsort(-mag[coords[:,0], coords[:,1]])]
    coords = coords[:max_points]
    points = []
    for y,x in coords:
        points.append((int(y), int(x), float(flow[y,x,0]), float(flow[y,x,1]), float(mag[y,x])))
    return {'dx_median': dx_med, 'dy_median': dy_med, 'points': points, 'num_moving': int(coords.shape[0])}


def apply_dsa(mask_frame, live_frame):
    """Custom DSA subtraction: live - mask*2, normalized safely with float32."""
    mask = mask_frame.astype(np.float32)
    live = live_frame.astype(np.float32)
    dsa = live - mask * 2.0
    dsa_shifted = dsa - np.min(dsa)
    range_val = np.max(dsa_shifted)
    if range_val == 0:
        range_val = 0.5
    dsa_normalized = (dsa_shifted / range_val) * 64.0
    return dsa_normalized.astype(np.uint8)


def save_uint8_png(path, img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

# -------------------------
# Visualization utility
# -------------------------

def visualize_alignment(template, frame, warp, motion, warped_aligned=None, dsa=None):
    h, w = template.shape[:2]
    aligned = warped_aligned if warped_aligned is not None else apply_warp(frame, warp, motion, (w,h))
    diff = (template - aligned)
    fig, ax = plt.subplots(1,4, figsize=(16,4))
    ax[0].imshow(template, cmap='gray'); ax[0].set_title('Template')
    ax[1].imshow(frame, cmap='gray'); ax[1].set_title('Input')
    ax[2].imshow(aligned, cmap='gray'); ax[2].set_title('Aligned')
    if dsa is None:
        ax[3].imshow(diff, cmap='seismic'); ax[3].set_title('Template - Aligned')
    else:
        ax[3].imshow(dsa, cmap='gray'); ax[3].set_title('DSA')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    dicom_path = "D:/Rohith/RAW_AUTOPIXEL/GI_RAW/1.2.826.0.1.3680043.2.1330.2641710.2312141403270002.5.464_Raw_anon.dcm"
    frame_idx_1 = 0
    frame_idx_2 = 29
    motion_model = cv2.MOTION_AFFINE
    pyr_levels = 1
    flow_threshold_pixels = 0.8
    max_motion_points = 300
    output_dir = os.path.join(os.getcwd(), "aps_outputs")
    os.makedirs(output_dir, exist_ok=True)

    frames = read_dicom_frames(dicom_path)
    print("Loaded frames:", frames.shape)
    T_raw = frames[frame_idx_1]
    I_raw = frames[frame_idx_2]

    dx, dy, resp = estimate_translation_phasecorr(T_raw, I_raw)
    print(f"PhaseCorr estimate -> dx={dx:.3f}, dy={dy:.3f}, response={resp:.4f}")

    ecc_val, warp = estimate_motion_ecc(T_raw, I_raw, motion=motion_model, pyr_levels=pyr_levels,
                                        number_of_iterations=500, termination_eps=1e-6, gaussFiltSize=5)
    print("ECC (correlation):", ecc_val)
    print("Warp matrix:\n", warp)

    params = decompose_warp2d(warp, motion_model)
    print("Decomposed params:", params)

    T = normalize_to_float32(T_raw)
    I = normalize_to_float32(I_raw)
    h, w = T.shape[:2]

    if warp is None:
        M = np.array([[1.0, 0.0, -dx],[0.0, 1.0, -dy]], dtype=np.float32)
        print("Using fallback translation affine:", M)
        warped_I = apply_warp(I, M, cv2.MOTION_AFFINE, (w,h))
    else:
        warped_I = apply_warp(I, warp, motion_model, (w,h))

    flow, mag = compute_optical_flow(T, I)
    summary = motion_summary_from_flow(flow, mag, threshold=flow_threshold_pixels, max_points=max_motion_points)
    print(f"Flow median dx={summary['dx_median']:.3f}, dy={summary['dy_median']:.3f}, moving points={summary['num_moving']}")

    csv_path = os.path.join(output_dir, 'motion_points.csv')
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['y','x','dx','dy','mag'])
        for p in summary['points']:
            writer.writerow(p)
    print('Wrote motion points ->', csv_path)

    dsa = apply_dsa(T_raw, warped_I)

    out_aligned = os.path.join(output_dir, 'aligned_live.png')
    out_dsa = os.path.join(output_dir, 'dsa_result.png')
    save_uint8_png(out_aligned, warped_I)
    save_uint8_png(out_dsa, dsa)
    print('Saved aligned live ->', out_aligned)
    print('Saved DSA ->', out_dsa)

    visualize_alignment(T_raw, I_raw, warp, motion_model, warped_aligned=warped_I, dsa=dsa)

    print('\nSummary:')
    print('  PhaseCorr dx,dy =', (dx, dy), 'resp=', resp)
    print('  ECC value =', ecc_val)
    print('  Warp params =', params)
    print('  Flow median dx,dy =', (summary['dx_median'], summary['dy_median']))
    print('  Moving points (sample):', summary['points'][:10])

    print('\nAll outputs are in', output_dir)