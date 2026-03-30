# GUI_9.PY - Post-processing removed
# Improved version with registration metrics export, ECC retry/fallback

import os
import numpy as np
import pydicom
import cv2
import logging
import csv
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter, QToolBar,
    QComboBox, QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMenu, QGroupBox
)
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('DSA')

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✅ GPU (CuPy) available - GPU acceleration ENABLED")
except Exception:
    GPU_AVAILABLE = False
    logger.info("⚠️ CuPy not available - Running in CPU mode")


# =========================================================
# RTX 4080 GPU Configuration
# =========================================================

class RTX4080Config:
    """
    Optimized settings for NVIDIA RTX 4080 processing
    1344×1344 resolution, 16-bit depth
    """
    
    GPU_ENABLED = True
    GPU_BATCH_SIZE = 4
    GPU_MEMORY_FRACTION = 0.85
    ENABLE_TENSOR_CORES = True
    
    IMAGE_WIDTH = 1344
    IMAGE_HEIGHT = 1344
    BIT_DEPTH = 16
    TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
    
    MOTION_THRESHOLD = 0.012
    MOTION_THRESHOLD_MIN = 0.005
    MOTION_THRESHOLD_MAX = 0.050
    
    BLEND_MARGIN = 18
    BLEND_MARGIN_FINE = 10
    BLEND_MARGIN_COARSE = 20
    
    ENABLE_LOCAL_REFINEMENT = True
    
    PCC_WINDOW_HANNING = True
    PCC_SUBPIXEL_PRECISION = True
    
    ECC_MOTION_TYPE = 'euclidean'
    ECC_PYRAMID_LEVELS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
    ECC_MAX_ITERATIONS = 7000
    ECC_EPSILON = 1e-9
    ECC_GAUSS_FILTER_SIZE = 7
    
    FLOW_METHOD = 'farneback'
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 6
    FLOW_WINSIZE = 35
    FLOW_ITERATIONS = 7
    FLOW_POLY_N = 7
    FLOW_POLY_SIGMA = 1.5
    FLOW_USE_INITIAL_FLOW = True
    
    GAUSSIAN_SIGMA = 3.0
    MORPH_KERNEL_SIZE = 11
    MORPH_CLOSE_ITERATIONS = 3
    MORPH_OPEN_ITERATIONS = 3
    
    ADAPTIVE_THRESHOLD = True
    USE_GRADIENT_INFO = True
    EDGE_PRESERVATION = True
    
    CLEAR_CACHE_EVERY_N_FRAMES = 10
    PREALLOCATE_BUFFERS = True
    USE_PINNED_MEMORY = True
    
    QUALITY_MODE = 'ultra'

# =========================================================
# GPU Memory Management
# =========================================================

def clear_gpu_cache():
    """Clear GPU memory cache."""
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()


def get_gpu_memory_info():
    """Check available GPU memory."""
    if not GPU_AVAILABLE:
        return
    mempool = cp.get_default_memory_pool()
    logger.info(f"GPU Memory Used: {mempool.used_bytes() / 1e9:.2f} GB")
    logger.info(f"GPU Memory Total: {mempool.total_bytes() / 1e9:.2f} GB")

# =========================================================
# Float Conversion Utilities
# =========================================================

def to_float(img, bit_depth=16):
    """Convert image to float32 in range [0, 1]"""
    max_val = float(2**bit_depth - 1)
    return img.astype(np.float32) / max_val


def from_float(img_float, bit_depth=16):
    """Convert float32 [0,1] image to uint16"""
    max_val = float(2**bit_depth - 1)
    return np.clip(img_float * max_val, 0, max_val).astype(np.uint16)

# =========================================================
# GPU-Accelerated Preprocessing Functions
# =========================================================

def gamma_correction_gpu(img_float, gamma):
    if GPU_AVAILABLE:
        img_gpu = cp.asarray(img_float)
        result = cp.clip(cp.power(img_gpu, gamma), 0, 1)
        return cp.asnumpy(result)
    else:
        return np.clip(np.power(img_float, gamma), 0, 1)

def apply_contrast_stretch(img_float):
    """Apply simple contrast stretching (min-max normalization)"""
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    stretched = (img_float - min_val) / (max_val - min_val + 1e-8)
    return np.clip(stretched, 0, 1)

def wiener_filter_gpu(img_float, kernel_size=5):
    if GPU_AVAILABLE:
        img_gpu = cp.asarray(img_float)
        kernel = cp.ones((kernel_size, kernel_size), cp.float32) / (kernel_size**2)
        from cupyx.scipy.ndimage import convolve
        local_mean = convolve(img_gpu, kernel, mode='reflect')
        local_mean_sq = convolve(img_gpu**2, kernel, mode='reflect')
        local_var = cp.maximum(local_mean_sq - local_mean**2, 1e-10)
        noise_var = float(cp.mean(local_var)) * 0.01
        gain = cp.clip((local_var - noise_var) / local_var, 0, 1)
        result = local_mean + gain * (img_gpu - local_mean)
        return cp.asnumpy(cp.clip(result, 0, 1))
    else:
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)
        local_mean_sq = cv2.filter2D(img_float**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
        local_var = np.maximum(local_mean_sq - local_mean**2, 1e-10)
        noise_var = np.mean(local_var) * 0.01
        gain = np.clip((local_var - noise_var) / local_var, 0, 1)
        result = local_mean + gain * (img_float - local_mean)
        return np.clip(result, 0, 1)

def apply_windowing_gpu(img_float, WL=0.4, WW=0.6):
    if GPU_AVAILABLE:
        img_gpu = cp.asarray(img_float)
        img_min = WL - WW / 2
        img_max = WL + WW / 2
        windowed = cp.clip(img_gpu, img_min, img_max)
        result = cp.clip((windowed - img_min) / (img_max - img_min), 0, 1)
        return cp.asnumpy(result)
    else:
        img_min = WL - WW / 2
        img_max = WL + WW / 2
        windowed = np.clip(img_float, img_min, img_max)
        return np.clip((windowed - img_min) / (img_max - img_min), 0, 1)

def apply_unsharp_mask_gpu(img_float, kernel_size=(5,5), sigma=1.0, amount=1.5):
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    sharpened = (amount + 1) * img_float - amount * blurred
    return np.clip(sharpened, 0, 1)

# =========================================================
# Preprocessing Pipeline
# =========================================================

def process_frame(frame, gamma, wiener_kernel=5, WL=0.4, WW=0.6):
    frame_float = to_float(frame)
    gm = gamma_correction_gpu(frame_float, gamma=gamma)
    cs = apply_contrast_stretch(gm)
    unsharp = apply_unsharp_mask_gpu(cs)
    win = apply_windowing_gpu(unsharp, WL=WL, WW=WW)
    wi = wiener_filter_gpu(win, kernel_size=wiener_kernel) 
    return wi

# =========================================================
# Histogram Matching
# =========================================================

def histogram_matching_gpu(source, reference, bins=256):
    DICOM_PIXEL_MAX = 65535.0
    if GPU_AVAILABLE:
        source_gpu = cp.asarray(source.astype(np.float32))
        reference_gpu = cp.asarray(reference.astype(np.float32))
        src_hist, src_bins = cp.histogram(source_gpu.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))
        ref_hist, ref_bins = cp.histogram(reference_gpu.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))
        src_hist = src_hist.astype(cp.float32) + 1e-6
        ref_hist = ref_hist.astype(cp.float32) + 1e-6
        src_cdf = cp.cumsum(src_hist)
        ref_cdf = cp.cumsum(ref_hist)
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        src_centers = src_bins[:-1] + (src_bins[1] - src_bins[0]) / 2.0
        ref_centers = ref_bins[:-1] + (ref_bins[1] - ref_bins[0]) / 2.0
        lookup_table = cp.interp(src_cdf, ref_cdf, ref_centers)
        indices = cp.searchsorted(src_centers, source_gpu.ravel(), side='left')
        indices = cp.clip(indices, 0, len(lookup_table) - 1)
        matched = lookup_table[indices].reshape(source_gpu.shape)
        result = cp.clip(matched, 0, DICOM_PIXEL_MAX).astype(cp.float32)
        return cp.asnumpy(result)
    else:
        source_uint16 = source.astype(np.uint16)
        reference_uint16 = reference.astype(np.uint16)
        src_hist, src_bins = np.histogram(source_uint16.ravel(), bins=bins, range=(0, int(DICOM_PIXEL_MAX)))
        ref_hist, ref_bins = np.histogram(reference_uint16.ravel(), bins=bins, range=(0, int(DICOM_PIXEL_MAX)))
        src_cdf = np.cumsum(src_hist).astype(np.float32)
        ref_cdf = np.cumsum(ref_hist).astype(np.float32)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]
        lookup_table = np.interp(src_cdf, ref_cdf, ref_bins[:-1])
        matched = np.interp(source_uint16.ravel(), src_bins[:-1], lookup_table)
        return matched.reshape(source.shape).astype(np.float32)

# =========================================================
# Phase Correlation Registration
# =========================================================

def _prep_for_fft(img_gpu, use_gpu=True):
    if use_gpu and GPU_AVAILABLE:
        img = img_gpu.astype(cp.float32)
        h, w = img.shape
        hann_y = cp.hanning(h).reshape(h, 1)
        hann_x = cp.hanning(w).reshape(1, w)
        window = hann_y * hann_x
        windowed = img * window
        return windowed - windowed.mean()
    else:
        img = img_gpu.astype(np.float32)
        h, w = img.shape
        hann_y = np.hanning(h).reshape(h, 1)
        hann_x = np.hanning(w).reshape(1, w)
        window = hann_y * hann_x
        windowed = img * window
        return windowed - windowed.mean()


def _parabolic_offset_1d(fm1, f0, fp1):
    denom = (fm1 - 2.0 * f0 + fp1)
    if abs(denom) < 1e-10:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom


def phase_correlation_registration(mask_frame, live_frame):
    """
    Compute phase correlation shift between two images with GPU acceleration.
    Returns (dy, dx, peak_value) - translation shifts and peak correlation value as confidence.
    """
    try:
        if GPU_AVAILABLE:
            mask_gpu = cp.asarray(mask_frame.astype(np.float32))
            live_gpu = cp.asarray(live_frame.astype(np.float32))
            a = _prep_for_fft(mask_gpu, use_gpu=True)
            b = _prep_for_fft(live_gpu, use_gpu=True)
            F1 = cp.fft.fft2(a)
            F2 = cp.fft.fft2(b)
            R = F1 * cp.conj(F2)
            R_mag = cp.abs(R)
            R = R / (R_mag + 1e-10)
            corr = cp.fft.ifft2(R).real
            iy, ix = map(int, cp.unravel_index(cp.argmax(corr), corr.shape))
            peak = float(cp.max(corr))
            h, w = corr.shape
            ym1, yp1 = (iy - 1) % h, (iy + 1) % h
            xm1, xp1 = (ix - 1) % w, (ix + 1) % w
            f0 = float(corr[iy, ix])
            off_y = _parabolic_offset_1d(float(corr[ym1, ix]), f0, float(corr[yp1, ix]))
            off_x = _parabolic_offset_1d(float(corr[iy, xm1]), f0, float(corr[iy, xp1]))
            int_y = iy if iy <= h // 2 else iy - h
            int_x = ix if ix <= w // 2 else ix - w
            dy = -(int_y + off_y)
            dx = -(int_x + off_x)
            return float(dy), float(dx), float(peak)
        else:
            a = _prep_for_fft(mask_frame.astype(np.float32), use_gpu=False)
            b = _prep_for_fft(live_frame.astype(np.float32), use_gpu=False)
            F1 = np.fft.fft2(a)
            F2 = np.fft.fft2(b)
            R = F1 * np.conj(F2)
            R_mag = np.abs(R)
            R = R / (R_mag + 1e-10)
            corr = np.fft.ifft2(R).real
            iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
            peak = float(np.max(corr))
            h, w = corr.shape
            ym1, yp1 = (iy - 1) % h, (iy + 1) % h
            xm1, xp1 = (ix - 1) % w, (ix + 1) % w
            f0 = float(corr[iy, ix])
            off_y = _parabolic_offset_1d(float(corr[ym1, ix]), f0, float(corr[yp1, ix]))
            off_x = _parabolic_offset_1d(float(corr[iy, xm1]), f0, float(corr[iy, xp1]))
            int_y = iy if iy <= h // 2 else iy - h
            int_x = ix if ix <= w // 2 else ix - w
            dy = -(int_y + off_y)
            dx = -(int_x + off_x)
            return float(dy), float(dx), float(peak)
    except Exception as e:
        logger.exception("Phase correlation failed")
        return 0.0, 0.0, 0.0

# =========================================================
# ECC Registration
# =========================================================

def _extract_rotation_and_scale_from_matrix(M):
    a, b, tx = M[0]
    c, d, ty = M[1]
    angle = np.degrees(np.arctan2(c, a))
    scale_x = np.sqrt(a*a + c*c)
    scale_y = np.sqrt(b*b + d*d)
    return angle, scale_x, scale_y


def ecc_registration(mask_frame, live_frame, levels=(0.25, 0.5, 1.0),
                     motionType=cv2.MOTION_EUCLIDEAN, retries=2):
    """
    Multi-scale ECC registration with retries and fallback.
    Returns (warp_matrix, ecc_score, method_used)
    """
    ref = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov = cv2.normalize(live_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ref = cv2.GaussianBlur(ref, (5, 5), 1.0)
    mov = cv2.GaussianBlur(mov, (5, 5), 1.0)

    base_levels = list(levels)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3000, 1e-7)

    method_used = 'euclidean'
    ecc_score = 0.0

    attempts = [cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE]

    for attempt in attempts:
        wm = np.eye(2, 3, dtype=np.float32)
        try:
            for scale in base_levels:
                ref_small = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                mov_small = cv2.resize(mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                wm_small = wm.copy()
                wm_small[:2, 2] *= scale
                retval, wm_small = cv2.findTransformECC(
                    ref_small,
                    mov_small,
                    wm_small,
                    motionType=attempt,
                    criteria=criteria,
                    inputMask=None,
                    gaussFiltSize=5
                )
                wm = wm_small.copy()
                wm[:2, 2] /= scale
                ecc_score = float(retval)

            warp_matrix = wm.copy()
            method_used = 'euclidean' if attempt == cv2.MOTION_EUCLIDEAN else 'affine'
            break

        except cv2.error as e:
            logger.warning(f"ECC attempt {attempt} failed: {e}")
            continue

    if ecc_score <= 0.2:
        try:
            logger.info("ECC score low, attempting optical flow fallback")
            flow = cv2.calcOpticalFlowFarneback(ref, mov, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            h, w = ref.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[...,0]).astype(np.float32)
            map_y = (grid_y + flow[...,1]).astype(np.float32)
            warped = cv2.remap(live_frame.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            warp_matrix = np.eye(2,3, dtype=np.float32)
            method_used = 'optical_flow'
            ecc_score = 0.0
            warp_matrix._warped_image = warped
        except Exception as e:
            logger.exception("Optical flow fallback failed")

    return warp_matrix, float(ecc_score), method_used

# =========================================================
# Utility warping functions
# =========================================================

def apply_translation_warp(img_float, dy, dx):
    h, w = img_float.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    warped = cv2.warpAffine(img_float, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0, 1)


def apply_affine_warp(img_float, transform_matrix):
    h, w = img_float.shape
    transform_matrix = transform_matrix.astype(np.float32)
    if hasattr(transform_matrix, '_warped_image'):
        warped = transform_matrix._warped_image
        return np.clip(warped, 0, 1)
    registered = cv2.warpAffine(img_float, transform_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(registered, 0, 1)

# =========================================================
# Hybrid registration
# =========================================================

def hybrid_registration(mask_frame, live_frame):
    dy, dx, peak = phase_correlation_registration(mask_frame, live_frame)
    initial_matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    M, ecc_score, method_used = ecc_registration(mask_frame, live_frame)
    if method_used == 'optical_flow':
        return M, {'dx': dx, 'dy': dy, 'ecc_score': ecc_score, 'method': method_used, 'pc_peak': peak}
    M[:2, 2] += initial_matrix[:2, 2]
    angle, sx, sy = _extract_rotation_and_scale_from_matrix(M)
    return M, {'dx': dx, 'dy': dy, 'ecc_score': ecc_score, 'angle': angle, 'scale_x': sx, 'scale_y': sy, 'method': method_used, 'pc_peak': peak}

def farneback_flow_registration_gpu(mask_frame, live_frame):
    """Local dense motion registration using GPU Farneback optical flow"""
    try:
        mask_u8 = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        live_u8 = cv2.normalize(live_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if GPU_AVAILABLE:
            prev_gpu = cv2.cuda_GpuMat()
            next_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(mask_u8)
            next_gpu.upload(live_u8)
            of = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=5, pyrScale=0.5, fastPyramids=False,
                winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
            )
            flow_gpu = of.calc(prev_gpu, next_gpu, None)
            flow = flow_gpu.download()
        else:
            flow = cv2.calcOpticalFlowFarneback(mask_u8, live_u8, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = mask_u8.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(live_frame.astype(np.float32), map_x, map_y,
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return warped, {'method': 'farneback_gpu', 'local_motion': True}

    except Exception as e:
        logger.exception("Farneback GPU registration failed")
        return live_frame, {'method': 'farneback_gpu', 'local_motion': False}


def tvl1_flow_registration_gpu(mask_frame, live_frame):
    """Local dense motion registration using GPU TV-L1 optical flow"""
    try:
        mask_u8 = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        live_u8 = cv2.normalize(live_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if GPU_AVAILABLE:
            prev_gpu = cv2.cuda_GpuMat()
            next_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(mask_u8)
            next_gpu.upload(live_u8)
            tvl1 = cv2.cuda_OpticalFlowDual_TVL1.create()
            flow_gpu = tvl1.calc(prev_gpu, next_gpu, None)
            flow = flow_gpu.download()
        else:
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = tvl1.calc(mask_u8, live_u8, None)

        h, w = mask_u8.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(live_frame.astype(np.float32), map_x, map_y,
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return warped, {'method': 'tvl1_gpu', 'local_motion': True}

    except Exception as e:
        logger.exception("TV-L1 GPU registration failed")
        return live_frame, {'method': 'tvl1_gpu', 'local_motion': False}

def calculate_similarity_transform_params(matrix):
    """Extract dx, dy, rotation, and uniform scale from 2x3 affine matrix."""
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    
    dx = float(tx)
    dy = float(ty)
    rotation = np.degrees(np.arctan2(c, a))
    scale = np.sqrt(a*a + c*c)
    
    return dx, dy, rotation, scale

def create_motion_mask_gpu_advanced(mask_frame, live_frame_warped, live_frame_original,
                                   threshold=0.015, gaussian_sigma=2.5, kernel_size=9,
                                   use_gradient=True):
    """Advanced motion mask creation with gradient-based edge detection"""
    try:
        if GPU_AVAILABLE:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter, binary_closing, binary_opening
            
            mask_gpu = cp.asarray(mask_frame)
            warped_gpu = cp.asarray(live_frame_warped)
            
            diff = cp.abs(mask_gpu - warped_gpu)
            
            if use_gradient:
                from cupyx.scipy.ndimage import sobel
                grad_y = sobel(diff, axis=0)
                grad_x = sobel(diff, axis=1)
                gradient_mag = cp.sqrt(grad_y**2 + grad_x**2)
                diff = diff + 0.3 * gradient_mag
            
            diff_smooth = gaussian_filter(diff, sigma=gaussian_sigma)
            motion_mask = (diff_smooth > threshold).astype(cp.uint8)
            
            kernel = cp.ones((kernel_size, kernel_size), cp.uint8)
            motion_mask = binary_closing(motion_mask, structure=kernel, iterations=2)
            motion_mask = binary_opening(motion_mask, structure=kernel, iterations=2)
            
            motion_mask_cpu = cp.asnumpy(motion_mask).astype(np.uint8)
            motion_magnitude_cpu = cp.asnumpy(diff_smooth).astype(np.float32)
            
        else:
            diff = np.abs(mask_frame - live_frame_warped)
            
            if use_gradient:
                grad_y = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
                grad_x = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
                gradient_mag = np.sqrt(grad_y**2 + grad_x**2)
                diff = diff + 0.3 * gradient_mag
            
            diff_smooth = cv2.GaussianBlur(diff, (2*kernel_size+1, 2*kernel_size+1), gaussian_sigma)
            motion_mask = (diff_smooth > threshold).astype(np.uint8)
            
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            motion_mask_cpu = motion_mask
            motion_magnitude_cpu = diff_smooth.astype(np.float32)
        
        return motion_mask_cpu, motion_magnitude_cpu
        
    except Exception as e:
        logger.exception("Advanced motion mask creation failed")
        return np.zeros(mask_frame.shape, dtype=np.uint8), np.zeros(mask_frame.shape, dtype=np.float32)

def apply_optical_flow_refinement_gpu(mask_frame, warped_frame, motion_mask, config):
    """Apply GPU-accelerated optical flow refinement only to moving regions"""
    try:
        mask_u8 = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        warped_u8 = cv2.normalize(warped_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if GPU_AVAILABLE and hasattr(cv2, 'cuda_FarnebackOpticalFlow'):
            prev_gpu = cv2.cuda_GpuMat()
            next_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(mask_u8)
            next_gpu.upload(warped_u8)
            
            of = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=config.FLOW_LEVELS,
                pyrScale=config.FLOW_PYR_SCALE,
                fastPyramids=False,
                winSize=config.FLOW_WINSIZE,
                numIters=config.FLOW_ITERATIONS,
                polyN=config.FLOW_POLY_N,
                polySigma=config.FLOW_POLY_SIGMA,
                flags=0
            )
            flow_gpu = of.calc(prev_gpu, next_gpu, None)
            flow = flow_gpu.download()
        else:
            flow = cv2.calcOpticalFlowFarneback(
                mask_u8, warped_u8, None,
                pyr_scale=config.FLOW_PYR_SCALE,
                levels=config.FLOW_LEVELS,
                winsize=config.FLOW_WINSIZE,
                iterations=config.FLOW_ITERATIONS,
                poly_n=config.FLOW_POLY_N,
                poly_sigma=config.FLOW_POLY_SIGMA,
                flags=0
            )
        
        h, w = mask_frame.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        motion_weight = motion_mask.astype(np.float32)
        map_x = (grid_x + flow[..., 0] * motion_weight).astype(np.float32)
        map_y = (grid_y + flow[..., 1] * motion_weight).astype(np.float32)
        
        warped_refined = cv2.remap(
            warped_frame,
            map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped_refined
        
    except Exception as e:
        logger.exception("Optical flow refinement failed")
        return warped_frame

def selective_pixel_transfer_advanced(mask_frame, live_frame_original, live_frame_warped,
                                     motion_mask, blend_margin=15, edge_preservation=True):
    """Advanced selective pixel transfer with edge preservation"""
    try:
        motion_mask_float = motion_mask.astype(np.float32)
        
        if edge_preservation:
            edges = cv2.Canny((motion_mask * 255).astype(np.uint8), 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edge_mask = (edges_dilated > 0).astype(np.float32)
            blend_margin_adaptive = blend_margin + edge_mask * blend_margin * 0.5
        else:
            blend_margin_adaptive = blend_margin
        
        if blend_margin > 0:
            dist_transform = cv2.distanceTransform(motion_mask, cv2.DIST_L2, 5)
            
            if edge_preservation:
                blend_weight = np.clip(dist_transform / (blend_margin + edge_mask * 5), 0, 1)
            else:
                blend_weight = np.clip(dist_transform / blend_margin, 0, 1)
            
            blend_weight = cv2.bilateralFilter(blend_weight.astype(np.float32), 9, 75, 75)
        else:
            blend_weight = motion_mask_float
        
        result_frame = (blend_weight * live_frame_warped + 
                       (1 - blend_weight) * mask_frame)
        
        return np.clip(result_frame, 0, 1).astype(np.float32)
        
    except Exception as e:
        logger.exception("Advanced pixel transfer failed")
        return live_frame_warped
    
def multiscale_motion_selective_registration_rtx4080(mask_frame, live_frame, config=None):
    """RTX 4080 Optimized Multi-Scale Registration"""
    
    if config is None:
        config = RTX4080Config()
    
    try:
        logger.info("🚀 RTX 4080 Multi-Scale Registration (Ultra Quality Mode)")
        
        if mask_frame.dtype == np.uint16:
            mask_float = to_float(mask_frame, bit_depth=config.BIT_DEPTH)
        else:
            mask_float = mask_frame.astype(np.float32)
            
        if live_frame.dtype == np.uint16:
            live_float = to_float(live_frame, bit_depth=config.BIT_DEPTH)
        else:
            live_float = live_frame.astype(np.float32)
        
        if mask_float.shape != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
            logger.warning(f"Image size {mask_float.shape} doesn't match config")
        
        logger.info(f"  Level 1: Phase Correlation (GPU, {config.IMAGE_WIDTH}×{config.IMAGE_HEIGHT})...")
        
        mask_uint16 = from_float(mask_float, bit_depth=config.BIT_DEPTH)
        live_uint16 = from_float(live_float, bit_depth=config.BIT_DEPTH)
        
        dy_pcc, dx_pcc, peak = phase_correlation_registration(mask_uint16, live_uint16)
        logger.info(f"    ✓ Phase Correlation: dx={dx_pcc:.3f}, dy={dy_pcc:.3f}, confidence={peak:.4f}")
        
        logger.info(f"  Level 2: ECC Registration ({len(config.ECC_PYRAMID_LEVELS)}-level pyramid)...")
        
        motion_type = cv2.MOTION_EUCLIDEAN if config.ECC_MOTION_TYPE == 'euclidean' else cv2.MOTION_AFFINE
        
        M_ecc, ecc_score, method_used = ecc_registration(
            mask_float,
            live_float,
            levels=config.ECC_PYRAMID_LEVELS,
            motionType=motion_type,
            retries=2
        )
        
        M_ecc[0, 2] += dx_pcc
        M_ecc[1, 2] += dy_pcc
        
        dx, dy, rotation, scale = calculate_similarity_transform_params(M_ecc)
        logger.info(f"    ✓ ECC: dx={dx:.3f}, dy={dy:.3f}, rot={rotation:.3f}°, scale={scale:.5f}")
        logger.info(f"    ✓ ECC Score: {ecc_score:.4f} ({method_used})")
        
        logger.info("  Level 3: Applying global transformation...")
        
        h, w = mask_float.shape
        live_warped_global = cv2.warpAffine(
            live_float,
            M_ecc,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        
        logger.info("  Level 4: Advanced motion detection (GPU-accelerated)...")
        
        if config.ADAPTIVE_THRESHOLD:
            img_std = np.std(mask_float - live_warped_global)
            adaptive_threshold = max(config.MOTION_THRESHOLD_MIN, 
                                   min(config.MOTION_THRESHOLD_MAX, 
                                       config.MOTION_THRESHOLD * (img_std / 0.1)))
            logger.info(f"    Adaptive threshold: {adaptive_threshold:.4f} (std={img_std:.4f})")
        else:
            adaptive_threshold = config.MOTION_THRESHOLD
        
        motion_mask, motion_magnitude = create_motion_mask_gpu_advanced(
            mask_float,
            live_warped_global,
            live_float,
            threshold=adaptive_threshold,
            gaussian_sigma=config.GAUSSIAN_SIGMA,
            kernel_size=config.MORPH_KERNEL_SIZE,
            use_gradient=config.USE_GRADIENT_INFO
        )
        
        num_moving_pixels = int(np.sum(motion_mask))
        motion_percentage = 100.0 * num_moving_pixels / config.TOTAL_PIXELS
        logger.info(f"    ✓ Motion detected: {num_moving_pixels:,}/{config.TOTAL_PIXELS:,} pixels ({motion_percentage:.2f}%)")
        
        if config.ENABLE_LOCAL_REFINEMENT and motion_percentage > 0.5:
            logger.info(f"  Level 5: Local refinement ({config.FLOW_METHOD.upper()} optical flow, GPU)...")
            
            try:
                live_warped_refined = apply_optical_flow_refinement_gpu(
                    mask_float,
                    live_warped_global,
                    motion_mask,
                    config
                )
                logger.info("    ✓ Local refinement applied successfully")
                
            except Exception as e:
                logger.warning(f"    ⚠ Local refinement failed: {e}, using global transform")
                live_warped_refined = live_warped_global
        else:
            live_warped_refined = live_warped_global
            if motion_percentage <= 0.5:
                logger.info("  Level 5: Skipped (motion < 0.5%)")
        
        logger.info("  Level 6: Edge-aware selective pixel transfer...")
        
        registered_frame = selective_pixel_transfer_advanced(
            mask_float,
            live_float,
            live_warped_refined,
            motion_mask,
            blend_margin=config.BLEND_MARGIN,
            edge_preservation=config.EDGE_PRESERVATION
        )
        
        logger.info("✅ RTX 4080 Multi-Scale Registration Complete!")
        
        metrics = {
            'method': 'multiscale_selective_rtx4080',
            'quality_mode': config.QUALITY_MODE,
            'dx': float(dx),
            'dy': float(dy),
            'rotation': float(rotation),
            'scale': float(scale),
            'ecc_score': float(ecc_score),
            'pc_peak': float(peak),
            'motion_pixels': num_moving_pixels,
            'motion_percentage': float(motion_percentage),
            'motion_threshold': float(adaptive_threshold),
            'image_resolution': f"{config.IMAGE_WIDTH}×{config.IMAGE_HEIGHT}",
            'bit_depth': config.BIT_DEPTH,
            'pyramid_levels': len(config.ECC_PYRAMID_LEVELS),
            'transform_matrix': M_ecc.tolist(),
            'registration_quality': 'excellent' if ecc_score > 0.9 else 'good' if ecc_score > 0.75 else 'acceptable'
        }
        
        return registered_frame, metrics
        
    except Exception as e:
        logger.exception("RTX 4080 Multi-Scale Registration failed")
        if live_frame.dtype == np.uint16:
            return to_float(live_frame), {'method': 'multiscale_selective_rtx4080', 'error': str(e)}
        else:
            return live_frame, {'method': 'multiscale_selective_rtx4080', 'error': str(e)}


# =========================================================
# DSA Subtraction (no post-processing)
# =========================================================

def perform_dsa(mask_frame, contrast_frame):
    dsa = contrast_frame - mask_frame
    dsa_min, dsa_max = dsa.min(), dsa.max()
    if dsa_max > dsa_min:
        dsa_normalized = (dsa - dsa_min) / (dsa_max - dsa_min)
    else:
        dsa_normalized = np.zeros_like(dsa)
    return np.clip(dsa_normalized, 0, 1)

# =========================================================
# Processing Thread
# =========================================================

class DSAProcessingThread(QThread):
    progress = Signal(int, str)
    finished = Signal(list, list, list)

    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames
        self.params = params
        self.registration_log = []

    def run(self):
        try:
            frames_preprocessed = []
            frames_registered = []
            frames_dsa = []

            self.progress.emit(10, "Preprocessing frames...")
            for i, frame in enumerate(self.frames):
                processed = process_frame(
                    frame,
                    gamma=self.params['gamma'],
                    wiener_kernel=self.params['wiener_kernel'],
                    WL=self.params['WL'],
                    WW=self.params['WW']
                )
                frames_preprocessed.append(processed)
                progress_pct = 10 + int(30 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Preprocessing frame {i+1}/{len(self.frames)}")

            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_preprocessed[mask_idx]
            mask_frame_uint16 = from_float(mask_frame)
            self.progress.emit(45, f"Using frame {mask_idx} as mask")

            self.progress.emit(50, "Registering frames...")
            registration_method = self.params['registration_method']

            for i, frame in enumerate(frames_preprocessed):
                if i == mask_idx:
                    frames_registered.append(frame)
                    self.registration_log.append({'frame': i, 'method': 'mask', 'dx': 0.0, 'dy': 0.0, 'angle': 0.0, 'ecc_score': 1.0})
                else:
                    frame_uint16 = from_float(frame)

                    if registration_method == 'pcc':
                        dy, dx, peak = phase_correlation_registration(mask_frame_uint16, frame_uint16)
                        registered = apply_translation_warp(frame, dy, dx)
                        log_entry = {'frame': i, 'method': 'pcc', 'dx': dx, 'dy': dy, 'angle': 0.0, 'ecc_score': peak}

                    elif registration_method == 'hybrid':
                        M, metrics = hybrid_registration(mask_frame, frame)
                        registered = apply_affine_warp(frame, M)
                        log_entry = {'frame': i, 'method': metrics.get('method', 'hybrid'),
                                     'dx': metrics.get('dx', 0.0), 'dy': metrics.get('dy', 0.0),
                                     'angle': metrics.get('angle', 0.0), 'ecc_score': metrics.get('ecc_score', 0.0)}
                    elif registration_method == 'farneback_gpu':
                        registered, metrics = farneback_flow_registration_gpu(mask_frame, frame)
                        log_entry = {'frame': i, **metrics}

                    elif registration_method == 'tvl1_gpu':
                        registered, metrics = tvl1_flow_registration_gpu(mask_frame, frame)
                        log_entry = {'frame': i, **metrics}

                    elif registration_method == 'multiscale_rtx4080':
                        registered, metrics = multiscale_motion_selective_registration_rtx4080(
                        mask_frame, 
                        frame,
                        config=RTX4080Config()
                    )
                        log_entry = {
                        'frame': i,
                        'method': metrics.get('method'),
                        'dx': metrics.get('dx', 0.0),
                        'dy': metrics.get('dy', 0.0),
                        'rotation': metrics.get('rotation', 0.0),
                        'scale': metrics.get('scale', 1.0),
                        'ecc_score': metrics.get('ecc_score', 0.0),
                        'motion_pixels': metrics.get('motion_pixels', 0),
                        'motion_percentage': metrics.get('motion_percentage', 0.0)
                    }
                        
                    else:
                        M, ecc_score, method_used = ecc_registration(mask_frame, frame)
                        registered = apply_affine_warp(frame, M)
                        angle, sx, sy = _extract_rotation_and_scale_from_matrix(M)
                        log_entry = {'frame': i, 'method': method_used, 'dx': M[0,2], 'dy': M[1,2], 'angle': angle, 'ecc_score': ecc_score}

                    if self.params['enable_histogram_matching']:
                        registered_uint16 = from_float(registered)
                        matched = histogram_matching_gpu(registered_uint16, mask_frame_uint16, bins=256)
                        registered = to_float(matched.astype(np.uint16))

                    frames_registered.append(registered)
                    self.registration_log.append(log_entry)

                progress_pct = 50 + int(30 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Registering frame {i+1}/{len(self.frames)}")

            frames_preprocessed = None

            try:
                out_csv = os.path.join(os.getcwd(), f"registration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                with open(out_csv, 'w', newline='') as csvfile:
                    fieldnames = ['frame', 'method', 'dx', 'dy', 'angle', 'ecc_score']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in self.registration_log:
                        writer.writerow({k: row.get(k, 0.0) for k in fieldnames})
                logger.info(f"Saved registration log: {out_csv}")
            except Exception:
                logger.exception("Failed to save registration log")

            self.progress.emit(85, "Performing DSA...")
            for i, contrast_frame in enumerate(frames_registered):
                if i == mask_idx:
                    dsa_result = np.zeros_like(mask_frame)
                else:
                    dsa_result = perform_dsa(mask_frame, contrast_frame)

                frames_dsa.append(dsa_result)
                progress_pct = 85 + int(15 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Processing DSA frame {i+1}/{len(self.frames)}")

            self.progress.emit(100, "Processing complete!")
            self.finished.emit([], frames_registered, frames_dsa)

            clear_gpu_cache()

        except Exception as e:
            self.progress.emit(0, f"Error: {str(e)}")
            logger.exception("Processing thread failed")

# =========================================================
# DICOM Save Utilities
# =========================================================

def save_multiframe_dicom(original_dcm, frames_float, out_path, description_suffix=""):
    frames_scaled = [from_float(f) for f in frames_float]
    new_dcm = FileDataset(out_path, {}, file_meta=original_dcm.file_meta, preamble=b"\0"*128)
    new_dcm.update(original_dcm)
    new_dcm.Rows, new_dcm.Columns = frames_scaled[0].shape
    new_dcm.NumberOfFrames = len(frames_scaled)
    new_dcm.SamplesPerPixel = 1
    new_dcm.PhotometricInterpretation = "MONOCHROME2"
    new_dcm.BitsAllocated = 16
    new_dcm.BitsStored = 16
    new_dcm.HighBit = 15
    new_dcm.PixelRepresentation = 0
    if description_suffix:
        original_desc = getattr(original_dcm, 'SeriesDescription', 'Unknown')
        new_dcm.SeriesDescription = f"{original_desc} {description_suffix}"
    new_dcm.SeriesInstanceUID = generate_uid()
    new_dcm.SOPInstanceUID = generate_uid()
    new_dcm.PixelData = np.stack(frames_scaled).tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    logger.info(f"✅ Saved multi-frame DICOM (16-bit): {out_path}")


def save_single_frame_dicom(original_dcm, frame_float, out_path, frame_number=0, description_suffix=""):
    frame_uint16 = from_float(frame_float)
    new_dcm = FileDataset(out_path, {}, file_meta=original_dcm.file_meta, preamble=b"\0"*128)
    new_dcm.update(original_dcm)
    new_dcm.Rows, new_dcm.Columns = frame_uint16.shape
    new_dcm.SamplesPerPixel = 1
    new_dcm.PhotometricInterpretation = "MONOCHROME2"
    new_dcm.BitsAllocated = 16
    new_dcm.BitsStored = 16
    new_dcm.HighBit = 15
    new_dcm.PixelRepresentation = 0
    new_dcm.InstanceNumber = frame_number + 1
    if description_suffix:
        original_desc = getattr(original_dcm, 'SeriesDescription', 'Unknown')
        new_dcm.SeriesDescription = f"{original_desc} {description_suffix} - Frame {frame_number}"
    new_dcm.SOPInstanceUID = generate_uid()
    if hasattr(new_dcm, 'NumberOfFrames'):
        delattr(new_dcm, 'NumberOfFrames')
    new_dcm.PixelData = frame_uint16.tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    logger.info(f"✅ Saved single-frame DICOM (16-bit): {out_path}")

# =========================================================
# DICOM Viewer Widget
# =========================================================

class DicomViewer(QWidget):
    def __init__(self, parent=None, dsa_mode=False):
        super().__init__(parent)
        self.frames = None
        self.frame_index = 0
        self.dsa_mode = dsa_mode
        self.original_dicom = None
        
        # Processing parameters
        self.gamma = 0.5
        self.wiener_kernel = 5
        self.WL = 0.4
        self.WW = 0.6
        self.mask_frame_index = 0
        self.registration_method = 'pcc'
        self.enable_histogram_matching = True
        
        # Processed frames cache
        self.frames_preprocessed = None
        self.frames_registered = None
        self.frames_dsa = None
        self.processing_thread = None

        layout = QVBoxLayout(self)
        self.filename_label = QLabel("No file loaded")
        self.info_label = QLabel("")
        layout.addWidget(self.filename_label)
        layout.addWidget(self.info_label)
        
        title = "DSA Viewer (GPU-Accelerated)" if dsa_mode else "Original DICOM"
        self.image_label = QLabel(title)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label, 1)
        
        if dsa_mode:
            self.setup_dsa_controls(layout)
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)
            
            self.save_btn = QPushButton("💾 Save DSA Results (DICOM)")
            self.save_btn.clicked.connect(self.show_save_menu)
            self.save_btn.setEnabled(False)
            layout.addWidget(self.save_btn)

    def setup_dsa_controls(self, layout):
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Preprocessing Parameters
        preprocess_group = QGroupBox("Preprocessing Parameters")
        preprocess_layout = QVBoxLayout()
        
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)  # 0.1 * 10
        self.gamma_slider.setMaximum(20)  # 2.0 * 10
        self.gamma_slider.setValue(5)  # 0.5 * 10
        self.gamma_slider.setTickPosition(QSlider.TicksBelow)
        self.gamma_slider.setTickInterval(1)
        self.gamma_label = QLabel("0.5")
        self.gamma_label.setMinimumWidth(40)
        self.gamma_slider.valueChanged.connect(lambda v: self.gamma_label.setText(f"{v/10:.1f}"))
        gamma_layout.addWidget(self.gamma_slider)
        gamma_layout.addWidget(self.gamma_label)
        preprocess_layout.addLayout(gamma_layout)
        
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("WL (Window Level):"))
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setMinimum(0)  # 0.0 * 20
        self.wl_slider.setMaximum(20)  # 1.0 * 20
        self.wl_slider.setValue(4)  # 0.2 * 20 (or use 8 for 0.4)
        self.wl_slider.setTickPosition(QSlider.TicksBelow)
        self.wl_slider.setTickInterval(1)
        self.wl_label = QLabel("0.20")
        self.wl_label.setMinimumWidth(40)
        self.wl_slider.valueChanged.connect(lambda v: self.wl_label.setText(f"{v/20:.2f}"))
        wl_layout.addWidget(self.wl_slider)
        wl_layout.addWidget(self.wl_label)
        preprocess_layout.addLayout(wl_layout)
        
        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("WW (Window Width):"))
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setMinimum(2)  # 0.1 * 20
        self.ww_slider.setMaximum(20)  # 1.0 * 20
        self.ww_slider.setValue(10)  # 0.5 * 20 (or use 12 for 0.6)
        self.ww_slider.setTickPosition(QSlider.TicksBelow)
        self.ww_slider.setTickInterval(1)
        self.ww_label = QLabel("0.50")
        self.ww_label.setMinimumWidth(40)
        self.ww_slider.valueChanged.connect(lambda v: self.ww_label.setText(f"{v/20:.2f}"))
        ww_layout.addWidget(self.ww_slider)
        ww_layout.addWidget(self.ww_label)
        preprocess_layout.addLayout(ww_layout)
    
        preprocess_group.setLayout(preprocess_layout)
        controls_layout.addWidget(preprocess_group)
               
        # Registration Parameters
        registration_group = QGroupBox("Registration Parameters")
        registration_layout = QVBoxLayout()
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Registration Method:"))
        self.reg_combo = QComboBox()
        self.reg_combo.addItems([
            'PCC (Phase Correlation)',
            'ECC (Enhanced Correlation)',
            'Hybrid (PCC + ECC)',
            'Local Flow (Farneback GPU)',
            'Local Flow (TV-L1 GPU)'
        ])
        self.reg_combo.setCurrentText('PCC (Phase Correlation)')
        row2.addWidget(self.reg_combo)
        
        row2.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row2.addWidget(self.mask_spin)
        registration_layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        self.hist_match_checkbox = QCheckBox("Enable Histogram Matching (GPU)")
        self.hist_match_checkbox.setChecked(True)
        row3.addWidget(self.hist_match_checkbox)
        registration_layout.addLayout(row3)
        
        registration_group.setLayout(registration_layout)
        controls_layout.addWidget(registration_group)
        
        # Process Button
        self.process_btn = QPushButton("🔄 Process DSA Pipeline (GPU)")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        self.process_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        controls_layout.addWidget(self.process_btn)
        
        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        # GPU Status
        if GPU_AVAILABLE:
            gpu_label = QLabel("✅ GPU Acceleration: ENABLED")
            gpu_label.setStyleSheet("color: green;")
        else:
            gpu_label = QLabel("⚠️ GPU Acceleration: DISABLED (CPU mode)")
            gpu_label.setStyleSheet("color: orange;")
        controls_layout.addWidget(gpu_label)
        
        layout.addWidget(controls_widget)

    def show_save_menu(self):
        menu = QMenu(self)
        menu.addAction("💾 Save Current Frame (Single DICOM)", self.save_current_frame_dicom)
        menu.addSeparator()
        menu.addAction("💾 Save All DSA Frames (Multi-frame DICOM)", self.save_multiframe_dicom_dsa)
        menu.addSeparator()
        menu.addAction("💾 Save Registered Frames (Multi-frame DICOM)", self.save_multiframe_dicom_registered)
        menu.exec_(self.save_btn.mapToGlobal(self.save_btn.rect().bottomLeft()))

    def save_current_frame_dicom(self):
        if self.frames_dsa is None or self.original_dicom is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Current Frame as DICOM",
            f"dsa_frame_{self.frame_index:03d}.dcm",
            "DICOM Files (*.dcm)"
        )
        if filename:
            try:
                frame = self.frames_dsa[self.frame_index]
                save_single_frame_dicom(
                    self.original_dicom, frame, filename,
                    frame_number=self.frame_index,
                    description_suffix="DSA Result"
                )
                QMessageBox.information(self, "Success", f"Frame saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save frame:\n{str(e)}")

    def save_multiframe_dicom_dsa(self):
        if self.frames_dsa is None or self.original_dicom is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save DSA as Multi-frame DICOM",
            "dsa_results_multiframe.dcm",
            "DICOM Files (*.dcm)"
        )
        if filename:
            try:
                save_multiframe_dicom(
                    self.original_dicom, self.frames_dsa,
                    filename, description_suffix="DSA Results"
                )
                QMessageBox.information(
                    self, "Success",
                    f"Saved {len(self.frames_dsa)} frames to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save DICOM:\n{str(e)}")

    def save_multiframe_dicom_registered(self):
        if self.frames_registered is None or self.original_dicom is None:
            QMessageBox.warning(self, "No Data", "No registered frames available!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Registered as Multi-frame DICOM",
            "registered_multiframe.dcm",
            "DICOM Files (*.dcm)"
        )
        if filename:
            try:
                save_multiframe_dicom(
                    self.original_dicom, self.frames_registered,
                    filename, description_suffix="Registered"
                )
                QMessageBox.information(
                    self, "Success",
                    f"Saved {len(self.frames_registered)} frames to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save DICOM:\n{str(e)}")

    def set_frames(self, frames, filename="", original_dicom=None):
        self.frames = frames
        self.frame_index = 0
        self.original_dicom = original_dicom
        if self.dsa_mode and hasattr(self, 'mask_spin'):
            self.mask_spin.setRange(0, len(frames) - 1)
        if filename:
            self.filename_label.setText(filename)
        self.info_label.setText(f"{len(frames)} frames")
        self.frames_preprocessed = None
        self.frames_registered = None
        self.frames_dsa = None
        if self.dsa_mode and hasattr(self, 'save_btn'):
            self.save_btn.setEnabled(False)
        self.show_frame()

    def process_dsa_pipeline(self):
        if 'Multi-Scale' in self.reg_combo.currentText():
            reg_method = 'multiscale_rtx4080'  
        elif 'Hybrid' in self.reg_combo.currentText():
            reg_method = 'hybrid'
        elif 'PCC' in self.reg_combo.currentText():
            reg_method = 'pcc'
        elif 'Farneback' in self.reg_combo.currentText():
            reg_method = 'farneback_gpu'
        elif 'TV-L1' in self.reg_combo.currentText():
            reg_method = 'tvl1_gpu'
        else:
            reg_method = 'ecc'

        params = {
            'gamma': self.gamma_slider.value()/10.0,
            'wiener_kernel': 5,
            'WL': self.wl_slider.value()/40.0,
            'WW': self.ww_slider.value()/40.0,
            'mask_frame_index': self.mask_spin.value(),
            'registration_method': reg_method,
            'enable_histogram_matching': self.hist_match_checkbox.isChecked()
        }

        self.gamma = params['gamma']
        self.WL = params['WL']
        self.WW = params['WW']
        self.mask_frame_index = params['mask_frame_index']
        self.registration_method = params['registration_method']
        self.enable_histogram_matching = params['enable_histogram_matching']

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Status: Processing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        self.processing_thread = DSAProcessingThread(self.frames, params)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Status: {message}")

    def on_processing_finished(self, preprocessed, registered, dsa):
        self.frames_preprocessed = None
        self.frames_registered = registered
        self.frames_dsa = dsa
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_label.setText("Status: Processing Complete!")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.show_frame()

    def set_frame_index(self, idx):
        if self.frames is None:
            return
        self.frame_index = idx % len(self.frames)
        self.show_frame()

    def show_frame(self):
        if self.frames is None:
            return
        try:
            if self.dsa_mode and self.frames_dsa is not None:
                dsa_frame = self.frames_dsa[self.frame_index]
                dsa_uint8 = (dsa_frame * 255).astype(np.uint8)
                h, w = dsa_uint8.shape
                qimg = QImage(dsa_uint8.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            else:
                frame = self.frames[self.frame_index]
                img_array = self._rescale_to_uint8(frame)
                h, w = img_array.shape
                qimg = QImage(img_array.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.image_label.setText(f"Display error: {str(e)[:50]}...")

    def _rescale_to_uint8(self, arr):
        a = arr.astype(np.float32)
        amin, amax = np.min(a), np.max(a)
        if amax == amin:
            return np.zeros(a.shape, dtype=np.uint8)
        a = (a - amin) / (amax - amin) * 255.0
        return a.astype(np.uint8)

# =========================================================
# Main Window
# =========================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced DSA Viewer - GPU-Accelerated Pipeline")
        self.resize(1600, 900)
        self.setAcceptDrops(True)
        
        splitter = QSplitter(Qt.Horizontal)
        self.viewer1 = DicomViewer(dsa_mode=False)
        splitter.addWidget(self.viewer1)
        self.viewer2 = DicomViewer(dsa_mode=True)
        splitter.addWidget(self.viewer2)
        splitter.setSizes([800, 800])
        self.setCentralWidget(splitter)
        
        self.setup_toolbar()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        self.setup_statusbar()
        
        self.frames = None
        self.frame_index = 0
        self.play_speed = 100
        self.dicom_dataset = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith('.dcm'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith('.dcm'):
                    self.load_dicom_file(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def setup_toolbar(self):
        toolbar = QToolBar()
        
        self.open_btn = QPushButton("📁 Open DICOM")
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        
        self.open_btn.clicked.connect(self.open_dicom)
        self.play_btn.clicked.connect(self.play_cine)
        self.pause_btn.clicked.connect(self.pause_cine)
        self.stop_btn.clicked.connect(self.stop_cine)
        
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.play_btn)
        toolbar.addWidget(self.pause_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel("  Frame: "))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        toolbar.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0/0")
        toolbar.addWidget(self.frame_label)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel("  Speed: "))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.5x', '1x', '2x', '4x'])
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        toolbar.addWidget(self.speed_combo)
        
        self.addToolBar(toolbar)

    def setup_statusbar(self):
        self.status_time = QLabel()
        self.status_fps = QLabel()
        self.statusBar().addWidget(self.status_time)
        self.statusBar().addPermanentWidget(self.status_fps)

    def open_dicom(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "",
            "DICOM Files (*.dcm);;All Files (*)"
        )
        if filename:
            self.load_dicom_file(filename)

    def load_dicom_file(self, filename):
        try:
            ds = pydicom.dcmread(filename)
            self.dicom_dataset = ds
            
            if hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1:
                pixel_array = ds.pixel_array
                if len(pixel_array.shape) == 3:
                    frames = [pixel_array[i] for i in range(pixel_array.shape[0])]
                else:
                    frames = [pixel_array]
            else:
                frames = [ds.pixel_array]
            
            self.frames = frames
            self.frame_index = 0
            short_name = os.path.basename(filename)
            
            self.viewer1.set_frames(frames, short_name, original_dicom=ds)
            self.viewer2.set_frames(frames, short_name, original_dicom=ds)
            
            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self.update_frame_label()
            
            self.statusBar().showMessage(
                f"Loaded: {short_name} ({len(frames)} frames)", 3000
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM:\n{str(e)}")

    def play_cine(self):
        if self.frames:
            self.timer.start(self.play_speed)

    def pause_cine(self):
        self.timer.stop()

    def stop_cine(self):
        self.timer.stop()
        if self.frames:
            self.frame_index = 0
            self.frame_slider.setValue(0)
            self.update_viewers()

    def next_frame(self):
        if self.frames:
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.frame_slider.setValue(self.frame_index)
            self.update_viewers()

    def on_slider_changed(self, value):
        if self.frames:
            self.frame_index = value
            self.update_viewers()
            self.update_frame_label()

    def on_speed_changed(self, text):
        speed_map = {'0.5x': 200, '1x': 100, '2x': 50, '4x': 25}
        self.play_speed = speed_map.get(text, 100)
        if self.timer.isActive():
            self.timer.setInterval(self.play_speed)

    def update_viewers(self):
        self.viewer1.set_frame_index(self.frame_index)
        self.viewer2.set_frame_index(self.frame_index)

    def update_frame_label(self):
        if self.frames:
            self.frame_label.setText(f"{self.frame_index + 1}/{len(self.frames)}")

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Fusion')
    
    if GPU_AVAILABLE:
        logger.info("=" * 70)
        logger.info("GPU-ACCELERATED DSA PIPELINE")
        logger.info("=" * 70)
        get_gpu_memory_info()
    else:
        logger.info("=" * 70)
        logger.info("DSA PIPELINE - CPU MODE")
        logger.info(" Install CuPy for GPU acceleration")
        logger.info("=" * 70)
    
    window = MainWindow()
    window.show()
    app.exec()
    
    if GPU_AVAILABLE:
        clear_gpu_cache()