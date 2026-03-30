import os
import numpy as np
import pydicom
import cv2
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

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU (CuPy) available - GPU acceleration ENABLED")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - Running in CPU mode")

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
    print(f"GPU Memory Used: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"GPU Memory Total: {mempool.total_bytes() / 1e9:.2f} GB")

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
    """Apply gamma correction with GPU acceleration."""
    if GPU_AVAILABLE:
        img_gpu = cp.asarray(img_float)
        result = cp.clip(cp.power(img_gpu, gamma), 0, 1)
        return cp.asnumpy(result)
    else:
        return np.clip(np.power(img_float, gamma), 0, 1)

def apply_clahe_gpu(img_float, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE with GPU optimization."""
    img_uint16 = np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_uint16)
    return np.clip(clahe_img.astype(np.float32) / 65535, 0, 1)

def wiener_filter_gpu(img_float, kernel_size=5):
    """Apply Wiener filter with GPU acceleration."""
    if GPU_AVAILABLE:
        img_gpu = cp.asarray(img_float)
        kernel = cp.ones((kernel_size, kernel_size), cp.float32) / (kernel_size**2)
        
        # Use scipy.ndimage equivalent in cupyx
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
    """Apply windowing with GPU acceleration."""
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
    """Apply unsharp masking with GPU optimization."""
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    sharpened = (amount + 1) * img_float - amount * blurred
    return np.clip(sharpened, 0, 1)

# =========================================================
# Preprocessing Pipeline
# =========================================================

def process_frame(frame, gamma, clahe_clip=2.0, clahe_tiles=(8, 8), 
                 wiener_kernel=5, WL=0.4, WW=0.6):
    """
    Complete preprocessing pipeline with GPU acceleration:
    Wiener → Gamma → CLAHE → Unsharp Masking → Windowing
    """
    frame_float = to_float(frame)
    
    wi = wiener_filter_gpu(frame_float, kernel_size=wiener_kernel)
    gm = gamma_correction_gpu(wi, gamma=gamma)
    cl = apply_clahe_gpu(gm, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    unsharp = apply_unsharp_mask_gpu(cl)
    win = apply_windowing_gpu(unsharp, WL=WL, WW=WW)
    
    return win

# =========================================================
# GPU-Accelerated Histogram Matching (from PCC.py)
# =========================================================

def histogram_matching_gpu(source, reference, bins=256):
    """
    Match histogram of source to reference with GPU acceleration.
    Optimized for 16-bit data.
    """
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
        # CPU fallback using cv2
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
# Phase Correlation Registration (from PCC.py)
# =========================================================

def _prep_for_fft(img_gpu, use_gpu=True):
    """Prepare image for FFT: apply Hann window and zero-mean."""
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
    """1D parabola fitting for subpixel accuracy."""
    denom = (fm1 - 2.0 * f0 + fp1)
    if abs(denom) < 1e-10:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom

def phase_correlation_registration(mask_frame, live_frame):
    """
    Compute phase correlation shift between two images with GPU acceleration.
    Returns (dy, dx) - translation shifts in pixels.
    """
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
        
        return float(dy), float(dx)
    else:
        # CPU fallback
        a = _prep_for_fft(mask_frame.astype(np.float32), use_gpu=False)
        b = _prep_for_fft(live_frame.astype(np.float32), use_gpu=False)

        F1 = np.fft.fft2(a)
        F2 = np.fft.fft2(b)

        R = F1 * np.conj(F2)
        R_mag = np.abs(R)
        R = R / (R_mag + 1e-10)

        corr = np.fft.ifft2(R).real

        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
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
        
        return float(dy), float(dx)

def apply_translation_warp(img_float, dy, dx):
    """Apply translation warp using affine transformation."""
    h, w = img_float.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    warped = cv2.warpAffine(img_float, M, (w, h), 
                           flags=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0, 1)

# =========================================================
# ECC Registration (from GUI_^.py)
# =========================================================

def ecc_registration(mask_frame, live_frame, levels=(0.25, 0.5, 1.0)):
    """
    Multi-scale ECC registration (rotation + translation only).
    Robust against low contrast and synthetic data.
    """
    # Normalize to uint8
    ref = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov = cv2.normalize(live_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply mild blur to stabilize gradients
    ref = cv2.GaussianBlur(ref, (5, 5), 1.0)
    mov = cv2.GaussianBlur(mov, (5, 5), 1.0)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3000, 1e-7)

    try:
        for scale in levels:
            ref_small = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mov_small = cv2.resize(mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            wm = warp_matrix.copy()
            wm[:2, 2] *= scale

            _, wm = cv2.findTransformECC(
                ref_small,
                mov_small,
                wm,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5
            )
            warp_matrix = wm.copy()

    except cv2.error as e:
        print(f"⚠️ ECC registration failed: {e}")
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    return warp_matrix


def apply_affine_warp(img_float, transform_matrix):
    """Apply affine transformation to image."""
    h, w = img_float.shape
    transform_matrix = transform_matrix.astype(np.float32)
    registered = cv2.warpAffine(img_float, transform_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
    return np.clip(registered, 0, 1)

def hybrid_registration(mask_frame, live_frame):
    """Hybrid registration: PCC for translation, ECC for refinement."""
    # Step 1: Phase correlation
    dy, dx = phase_correlation_registration(mask_frame, live_frame)
    initial_matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)

    # Step 2: ECC refinement
    M = ecc_registration(mask_frame, live_frame)

    # Combine PCC translation with ECC rotation/translation
    M[:2, 2] += initial_matrix[:2, 2]
    return M


# =========================================================
# DSA Subtraction
# =========================================================

def perform_dsa(mask_frame, contrast_frame):
    """Perform Digital Subtraction Angiography with standard normalization"""
    dsa = contrast_frame - mask_frame
    
    dsa_min, dsa_max = dsa.min(), dsa.max()
    if dsa_max > dsa_min:
        dsa_normalized = (dsa - dsa_min) / (dsa_max - dsa_min)
    else:
        dsa_normalized = np.zeros_like(dsa)
    
    return np.clip(dsa_normalized, 0, 1)

# =========================================================
# Post-Processing Functions
# =========================================================

def post_process_dsa(dsa_frame, gamma=0.5, gaussian_sigma=0.6, 
                     nlm_h=0.0125, clahe_clip=2, clahe_tiles=(4,4),
                     unsharp_amount=0.45, unsharp_radius=0.35):
    """
    Apply post-processing pipeline to DSA result.
    """
    # Gamma correction
    img = gamma_correction_gpu(dsa_frame, gamma=gamma)
    
    # Gaussian denoising
    img_uint16 = (img * 65535).astype(np.uint16)
    img_denoised = cv2.GaussianBlur(img_uint16, (5, 5), gaussian_sigma)
    img = img_denoised.astype(np.float32) / 65535
    
    # NLM denoising
    img_uint8 = (img * 255).astype(np.uint8)
    h_scaled = int(nlm_h * 255)
    img_nlm = cv2.fastNlMeansDenoising(img_uint8, None, h=h_scaled, templateWindowSize=7, searchWindowSize=21)
    img = img_nlm.astype(np.float32) / 255
    
    # CLAHE enhancement
    img = apply_clahe_gpu(img, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    
    # Unsharp masking
    kernel_size = max(3, int(unsharp_radius * 10))
    if kernel_size % 2 == 0:
        kernel_size += 1
    img = apply_unsharp_mask_gpu(img, kernel_size=(kernel_size, kernel_size), 
                                 sigma=unsharp_radius, amount=unsharp_amount)
    
    return np.clip(img, 0, 1)

# =========================================================
# Processing Thread
# =========================================================

class DSAProcessingThread(QThread):
    """Background thread for DSA processing with GPU acceleration"""
    progress = Signal(int, str)
    finished = Signal(list, list, list)
    
    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames
        self.params = params
        
    def run(self):
        try:
            frames_preprocessed = []
            frames_registered = []
            frames_dsa = []
            
            # Step 1: Preprocess all frames
            self.progress.emit(10, "Preprocessing frames...")
            for i, frame in enumerate(self.frames):
                processed = process_frame(
                    frame,
                    gamma=self.params['gamma'],
                    clahe_clip=self.params['clahe_clip'],
                    clahe_tiles=self.params['clahe_tiles'],
                    wiener_kernel=self.params['wiener_kernel'],
                    WL=self.params['WL'],
                    WW=self.params['WW']
                )
                frames_preprocessed.append(processed)
                progress_pct = 10 + int(30 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Preprocessing frame {i+1}/{len(self.frames)}")
            
            # Step 2: Get mask frame
            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_preprocessed[mask_idx]
            mask_frame_uint16 = from_float(mask_frame)
            self.progress.emit(45, f"Using frame {mask_idx} as mask")
            
            # Step 3: Register frames
            self.progress.emit(50, "Registering frames...")
            registration_method = self.params['registration_method']
            
            for i, frame in enumerate(frames_preprocessed):
                if i == mask_idx:
                    frames_registered.append(frame)
                else:
                    frame_uint16 = from_float(frame)
                    
                    if registration_method == 'pcc':
                        dy, dx = phase_correlation_registration(mask_frame_uint16, frame_uint16)
                        registered = apply_translation_warp(frame, dy, dx)
                    elif registration_method == 'hybrid':
    # Step 1 – PCC translation
                        dy, dx = phase_correlation_registration(mask_frame_uint16, frame_uint16)
                        initial_transform = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    
    # Step 2 – ECC refinement
                        refine_transform = ecc_registration(mask_frame, frame)
    
    # Combine translations
                        refine_transform[:2, 2] += initial_transform[:2, 2]
                        registered = apply_affine_warp(frame, refine_transform)

                    else:  # ecc only
                        transform = ecc_registration(mask_frame, frame)
                        registered = apply_affine_warp(frame, transform)
                    
                    # Histogram matching if enabled
                    if self.params['enable_histogram_matching']:
                        registered_uint16 = from_float(registered)
                        matched = histogram_matching_gpu(registered_uint16, mask_frame_uint16, bins=256)
                        registered = to_float(matched.astype(np.uint16))
                    
                    frames_registered.append(registered)
                
                progress_pct = 50 + int(30 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Registering frame {i+1}/{len(self.frames)}")
            
            # Step 4: Perform DSA
            self.progress.emit(85, "Performing DSA...")
            for i, contrast_frame in enumerate(frames_registered):
                if i == mask_idx:
                    dsa_result = np.zeros_like(mask_frame)
                else:
                    dsa_result = perform_dsa(
                        mask_frame, contrast_frame)
                
                # Apply post-processing if enabled
                if self.params['enable_post_processing']:
                    dsa_result = post_process_dsa(
                        dsa_result,
                        gamma=self.params['post_gamma'],
                        gaussian_sigma=self.params['post_gaussian_sigma'],
                        nlm_h=self.params['post_nlm_h'],
                        clahe_clip=self.params['post_clahe_clip'],
                        clahe_tiles=self.params['post_clahe_tiles'],
                        unsharp_amount=self.params['post_unsharp_amount'],
                        unsharp_radius=self.params['post_unsharp_radius']
                    )
                
                frames_dsa.append(dsa_result)
                
                progress_pct = 85 + int(15 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Processing DSA frame {i+1}/{len(self.frames)}")
            
            self.progress.emit(100, "Processing complete!")
            self.finished.emit(frames_preprocessed, frames_registered, frames_dsa)
            
            # Clear GPU cache
            clear_gpu_cache()
            
        except Exception as e:
            self.progress.emit(0, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

# =========================================================
# DICOM Save Utilities
# =========================================================

def save_multiframe_dicom(original_dcm, frames_float, out_path, description_suffix=""):
    """Save frames as multi-frame DICOM with proper 16-bit encoding."""
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
    
    # Generate new UIDs
    new_dcm.SeriesInstanceUID = generate_uid()
    new_dcm.SOPInstanceUID = generate_uid()
    
    new_dcm.PixelData = np.stack(frames_scaled).tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    
    print(f"✅ Saved multi-frame DICOM (16-bit): {out_path}")

def save_single_frame_dicom(original_dcm, frame_float, out_path, frame_number=0, description_suffix=""):
    """Save single frame as DICOM."""
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
    
    # Generate new UIDs
    new_dcm.SOPInstanceUID = generate_uid()
    
    # Remove multi-frame attribute if exists
    if hasattr(new_dcm, 'NumberOfFrames'):
        delattr(new_dcm, 'NumberOfFrames')
    
    new_dcm.PixelData = frame_uint16.tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    
    print(f"✅ Saved single-frame DICOM (16-bit): {out_path}")

# =========================================================
# DICOM Viewer Widget
# =========================================================

class DicomViewer(QWidget):
    """DICOM viewer with GPU-accelerated DSA processing"""
    def __init__(self, parent=None, dsa_mode=False):
        super().__init__(parent)
        self.frames = None
        self.frame_index = 0
        self.dsa_mode = dsa_mode
        self.original_dicom = None
        
        # Processing parameters
        self.gamma = 0.5
        self.clahe_clip = 2.0
        self.clahe_tiles = (8, 8)
        self.wiener_kernel = 5
        self.WL = 0.4
        self.WW = 0.6
        self.mask_frame_index = 0
        self.registration_method = 'pcc'
        self.enable_histogram_matching = True
        self.enable_post_processing = False
        
        # Post-processing parameters
        self.post_gamma = 0.5
        self.post_gaussian_sigma = 0.6
        self.post_nlm_h = 0.0125
        self.post_clahe_clip = 2
        self.post_clahe_tiles = (4, 4)
        self.post_unsharp_amount = 0.45
        self.post_unsharp_radius = 0.35
        
        # Processed frames cache
        self.frames_preprocessed = None
        self.frames_registered = None
        self.frames_dsa = None
        self.processing_thread = None

        layout = QVBoxLayout(self)
        
        # File info
        self.filename_label = QLabel("No file loaded")
        self.info_label = QLabel("")
        layout.addWidget(self.filename_label)
        layout.addWidget(self.info_label)
        

# Image display
        title = "DSA Viewer (GPU-Accelerated)" if dsa_mode else "Original DICOM"
        self.image_label = QLabel(title)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label, 1)

        # DSA Controls
        if dsa_mode:
            self.setup_dsa_controls(layout)
            
            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)
            
            # Save button
            self.save_btn = QPushButton("💾 Save DSA Results (DICOM)")
            self.save_btn.clicked.connect(self.show_save_menu)
            self.save_btn.setEnabled(False)
            layout.addWidget(self.save_btn)

    def setup_dsa_controls(self, layout):
        """Setup DSA processing controls"""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # ========== PREPROCESSING GROUP ==========
        preprocess_group = QGroupBox("Preprocessing Parameters")
        preprocess_layout = QVBoxLayout()
        
        # Row 1: Gamma, WL, WW
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Gamma:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 2.0)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.setSingleStep(0.1)
        row1.addWidget(self.gamma_spin)
        
        row1.addWidget(QLabel("WL (Window Level):"))
        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(0.0, 1.0)
        self.wl_spin.setValue(0.4)
        self.wl_spin.setSingleStep(0.05)
        row1.addWidget(self.wl_spin)
        
        row1.addWidget(QLabel("WW (Window Width):"))
        self.ww_spin = QDoubleSpinBox()
        self.ww_spin.setRange(0.1, 1.0)
        self.ww_spin.setValue(0.6)
        self.ww_spin.setSingleStep(0.05)
        row1.addWidget(self.ww_spin)
        
        preprocess_layout.addLayout(row1)
        preprocess_group.setLayout(preprocess_layout)
        controls_layout.addWidget(preprocess_group)
        
        # ========== REGISTRATION GROUP ==========
        registration_group = QGroupBox("Registration Parameters")
        registration_layout = QVBoxLayout()
        
        # Row 2: Registration method and mask frame
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Registration Method:"))
        self.reg_combo = QComboBox()
        self.reg_combo.addItems(['PCC (Phase Correlation)', 'ECC (Enhanced Correlation)', 'Hybrid (PCC + ECC)'])
        self.reg_combo.setCurrentText('PCC (Phase Correlation)')
        row2.addWidget(self.reg_combo)
        
        row2.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row2.addWidget(self.mask_spin)
        
        registration_layout.addLayout(row2)
        
        # Row 3: Histogram matching
        row3 = QHBoxLayout()
        self.hist_match_checkbox = QCheckBox("Enable Histogram Matching (GPU)")
        self.hist_match_checkbox.setChecked(True)
        row3.addWidget(self.hist_match_checkbox)
        registration_layout.addLayout(row3)
        
        registration_group.setLayout(registration_layout)
        controls_layout.addWidget(registration_group)
                
        # ========== POST-PROCESSING GROUP ==========
        postprocess_group = QGroupBox("Post-Processing (Optional)")
        postprocess_layout = QVBoxLayout()
        
        # Enable post-processing checkbox
        self.postprocess_checkbox = QCheckBox("Enable Post-Processing Pipeline")
        self.postprocess_checkbox.setChecked(False)
        self.postprocess_checkbox.stateChanged.connect(self.toggle_postprocess_controls)
        postprocess_layout.addWidget(self.postprocess_checkbox)
        
        # Post-processing controls container
        self.postprocess_controls = QWidget()
        postprocess_controls_layout = QVBoxLayout()
        
        # Post-processing parameters
        post_row1 = QHBoxLayout()
        post_row1.addWidget(QLabel("Gamma:"))
        self.post_gamma_spin = QDoubleSpinBox()
        self.post_gamma_spin.setRange(0.1, 2.0)
        self.post_gamma_spin.setValue(0.5)
        self.post_gamma_spin.setSingleStep(0.1)
        post_row1.addWidget(self.post_gamma_spin)
        
        post_row1.addWidget(QLabel("Gaussian σ:"))
        self.post_gaussian_spin = QDoubleSpinBox()
        self.post_gaussian_spin.setRange(0.1, 2.0)
        self.post_gaussian_spin.setValue(0.6)
        self.post_gaussian_spin.setSingleStep(0.1)
        post_row1.addWidget(self.post_gaussian_spin)
        postprocess_controls_layout.addLayout(post_row1)
        
        post_row2 = QHBoxLayout()
        post_row2.addWidget(QLabel("NLM h:"))
        self.post_nlm_spin = QDoubleSpinBox()
        self.post_nlm_spin.setRange(0.001, 0.1)
        self.post_nlm_spin.setValue(0.0125)
        self.post_nlm_spin.setSingleStep(0.001)
        self.post_nlm_spin.setDecimals(4)
        post_row2.addWidget(self.post_nlm_spin)
        
        post_row2.addWidget(QLabel("CLAHE Clip:"))
        self.post_clahe_spin = QDoubleSpinBox()
        self.post_clahe_spin.setRange(1.0, 5.0)
        self.post_clahe_spin.setValue(2.0)
        self.post_clahe_spin.setSingleStep(0.5)
        post_row2.addWidget(self.post_clahe_spin)
        postprocess_controls_layout.addLayout(post_row2)
        
        post_row3 = QHBoxLayout()
        post_row3.addWidget(QLabel("Unsharp Amount:"))
        self.post_unsharp_amount_spin = QDoubleSpinBox()
        self.post_unsharp_amount_spin.setRange(0.0, 2.0)
        self.post_unsharp_amount_spin.setValue(0.45)
        self.post_unsharp_amount_spin.setSingleStep(0.05)
        post_row3.addWidget(self.post_unsharp_amount_spin)
        
        post_row3.addWidget(QLabel("Unsharp Radius:"))
        self.post_unsharp_radius_spin = QDoubleSpinBox()
        self.post_unsharp_radius_spin.setRange(0.1, 2.0)
        self.post_unsharp_radius_spin.setValue(0.35)
        self.post_unsharp_radius_spin.setSingleStep(0.05)
        post_row3.addWidget(self.post_unsharp_radius_spin)
        postprocess_controls_layout.addLayout(post_row3)
        
        self.postprocess_controls.setLayout(postprocess_controls_layout)
        self.postprocess_controls.setEnabled(False)
        postprocess_layout.addWidget(self.postprocess_controls)
        
        postprocess_group.setLayout(postprocess_layout)
        controls_layout.addWidget(postprocess_group)
        
        # ========== PROCESS BUTTON ==========
        self.process_btn = QPushButton("🔄 Process DSA Pipeline (GPU)")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        self.process_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        controls_layout.addWidget(self.process_btn)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        # GPU status
        if GPU_AVAILABLE:
            gpu_label = QLabel("✅ GPU Acceleration: ENABLED")
            gpu_label.setStyleSheet("color: green;")
        else:
            gpu_label = QLabel("⚠️ GPU Acceleration: DISABLED (CPU mode)")
            gpu_label.setStyleSheet("color: orange;")
        controls_layout.addWidget(gpu_label)
        
        layout.addWidget(controls_widget)

    def toggle_postprocess_controls(self, state):
        """Enable/disable post-processing controls"""
        self.postprocess_controls.setEnabled(state == Qt.Checked)

    def show_save_menu(self):
        """Show save options menu"""
        menu = QMenu(self)
        
        menu.addAction("💾 Save Current Frame (Single DICOM)", self.save_current_frame_dicom)
        menu.addSeparator()
        menu.addAction("💾 Save All DSA Frames (Multi-frame DICOM)", self.save_multiframe_dicom_dsa)
        menu.addSeparator()
        menu.addAction("💾 Save Preprocessed Frames (Multi-frame DICOM)", self.save_multiframe_dicom_preprocessed)
        menu.addAction("💾 Save Registered Frames (Multi-frame DICOM)", self.save_multiframe_dicom_registered)
        
        menu.exec_(self.save_btn.mapToGlobal(self.save_btn.rect().bottomLeft()))

    def save_current_frame_dicom(self):
        """Save current DSA frame as single DICOM"""
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
                    self.original_dicom, 
                    frame, 
                    filename, 
                    frame_number=self.frame_index,
                    description_suffix="DSA Result"
                )
                QMessageBox.information(self, "Success", f"Frame saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save frame:\n{str(e)}")

    def save_multiframe_dicom_dsa(self):
        """Save all DSA frames as multi-frame DICOM"""
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
                    self.original_dicom,
                    self.frames_dsa,
                    filename,
                    description_suffix="DSA Results"
                )
                QMessageBox.information(self, "Success", 
                    f"Saved {len(self.frames_dsa)} frames to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save DICOM:\n{str(e)}")

    def save_multiframe_dicom_preprocessed(self):
        """Save preprocessed frames as multi-frame DICOM"""
        if self.frames_preprocessed is None or self.original_dicom is None:
            QMessageBox.warning(self, "No Data", "No preprocessed frames available!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Preprocessed as Multi-frame DICOM",
            "preprocessed_multiframe.dcm",
            "DICOM Files (*.dcm)"
        )
        
        if filename:
            try:
                save_multiframe_dicom(
                    self.original_dicom,
                    self.frames_preprocessed,
                    filename,
                    description_suffix="Preprocessed"
                )
                QMessageBox.information(self, "Success", 
                    f"Saved {len(self.frames_preprocessed)} frames to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save DICOM:\n{str(e)}")

    def save_multiframe_dicom_registered(self):
        """Save registered frames as multi-frame DICOM"""
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
                    self.original_dicom,
                    self.frames_registered,
                    filename,
                    description_suffix="Registered"
                )
                QMessageBox.information(self, "Success", 
                    f"Saved {len(self.frames_registered)} frames to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save DICOM:\n{str(e)}")

    def set_frames(self, frames, filename="", original_dicom=None):
        """Set frames for viewing"""
        self.frames = frames
        self.frame_index = 0
        self.original_dicom = original_dicom
        
        # Update mask frame range
        if self.dsa_mode and hasattr(self, 'mask_spin'):
            self.mask_spin.setRange(0, len(frames) - 1)
        
        if filename:
            self.filename_label.setText(filename)
        self.info_label.setText(f"{len(frames)} frames")
        
        # Clear processed frames
        self.frames_preprocessed = None
        self.frames_registered = None
        self.frames_dsa = None
        
        if self.dsa_mode and hasattr(self, 'save_btn'):
            self.save_btn.setEnabled(False)
        
        self.show_frame()

    def process_dsa_pipeline(self):
        """Start DSA processing in background thread"""
        if self.frames is None:
            return
        
        # Get parameters
        if 'Hybrid' in self.reg_combo.currentText():
            reg_method = 'hybrid'
        elif 'PCC' in self.reg_combo.currentText():
            reg_method = 'pcc'
        else:
            reg_method = 'ecc'

        params = {
            'gamma': self.gamma_spin.value(),
            'clahe_clip': 2.0,  
            'clahe_tiles': (8, 8),  
            'wiener_kernel': 5,  
            'WL': self.wl_spin.value(),
            'WW': self.ww_spin.value(),
            'mask_frame_index': self.mask_spin.value(),
            'registration_method': reg_method,
            'enable_histogram_matching': self.hist_match_checkbox.isChecked(),
            'enable_post_processing': self.postprocess_checkbox.isChecked(),
            'post_gamma': self.post_gamma_spin.value(),
            'post_gaussian_sigma': self.post_gaussian_spin.value(),
            'post_nlm_h': self.post_nlm_spin.value(),
            'post_clahe_clip': int(self.post_clahe_spin.value()),
            'post_clahe_tiles': (4, 4),
            'post_unsharp_amount': self.post_unsharp_amount_spin.value(),
            'post_unsharp_radius': self.post_unsharp_radius_spin.value()
        }
        
        # Update stored parameters
        self.gamma = params['gamma']
        self.WL = params['WL']
        self.WW = params['WW']
        self.mask_frame_index = params['mask_frame_index']
        self.registration_method = params['registration_method']
        self.enable_histogram_matching = params['enable_histogram_matching']
        self.enable_post_processing = params['enable_post_processing']
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Status: Processing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # Start processing thread
        self.processing_thread = DSAProcessingThread(self.frames, params)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Status: {message}")

    def on_processing_finished(self, preprocessed, registered, dsa):
        """Handle processing completion"""
        self.frames_preprocessed = preprocessed
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
                # Show DSA result in grayscale
                dsa_frame = self.frames_dsa[self.frame_index]
                
                # Convert to uint8 grayscale
                dsa_uint8 = (dsa_frame * 255).astype(np.uint8)
                
                h, w = dsa_uint8.shape
                qimg = QImage(dsa_uint8.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            else:
                # Show original frame
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
        
        # Enable drag and drop
        self.setAcceptDrops(True)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Original viewer
        self.viewer1 = DicomViewer(dsa_mode=False)
        splitter.addWidget(self.viewer1)
        
        # DSA viewer
        self.viewer2 = DicomViewer(dsa_mode=True)
        splitter.addWidget(self.viewer2)
        
        splitter.setSizes([800, 800])
        self.setCentralWidget(splitter)

        # Toolbar
        self.setup_toolbar()

        # Timer for cine loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Status bar
        self.setup_statusbar()

        self.frames = None
        self.frame_index = 0
        self.play_speed = 100
        self.dicom_dataset = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith('.dcm'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
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
        """Open DICOM file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if filename:
            self.load_dicom_file(filename)

    def load_dicom_file(self, filename):
        """Load DICOM file and extract frames"""
        try:
            ds = pydicom.dcmread(filename)
            self.dicom_dataset = ds
            
            # Extract frames
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
            
            # Update viewers
            short_name = os.path.basename(filename)
            self.viewer1.set_frames(frames, short_name, original_dicom=ds)
            self.viewer2.set_frames(frames, short_name, original_dicom=ds)
            
            # Update slider
            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self.update_frame_label()
            
            self.statusBar().showMessage(f"Loaded: {short_name} ({len(frames)} frames)", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM:\n{str(e)}")

    def play_cine(self):
        """Start cine loop"""
        if self.frames:
            self.timer.start(self.play_speed)

    def pause_cine(self):
        """Pause cine loop"""
        self.timer.stop()

    def stop_cine(self):
        """Stop cine loop and reset to first frame"""
        self.timer.stop()
        if self.frames:
            self.frame_index = 0
            self.frame_slider.setValue(0)
            self.update_viewers()

    def next_frame(self):
        """Advance to next frame"""
        if self.frames:
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.frame_slider.setValue(self.frame_index)
            self.update_viewers()

    def on_slider_changed(self, value):
        """Handle slider change"""
        if self.frames:
            self.frame_index = value
            self.update_viewers()
            self.update_frame_label()

    def on_speed_changed(self, text):
        """Handle speed change"""
        speed_map = {'0.5x': 200, '1x': 100, '2x': 50, '4x': 25}
        self.play_speed = speed_map.get(text, 100)
        if self.timer.isActive():
            self.timer.setInterval(self.play_speed)

    def update_viewers(self):
        """Update both viewers"""
        self.viewer1.set_frame_index(self.frame_index)
        self.viewer2.set_frame_index(self.frame_index)

    def update_frame_label(self):
        """Update frame label"""
        if self.frames:
            self.frame_label.setText(f"{self.frame_index + 1}/{len(self.frames)}")

# =========================================================
# Main Entry Point
# =========================================================

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Fusion')
    
    # Print GPU status
    if GPU_AVAILABLE:
        print("=" * 70)
        print("GPU-ACCELERATED DSA PIPELINE")
        print("=" * 70)
        get_gpu_memory_info()
    else:
        print("=" * 70)
        print("DSA PIPELINE - CPU MODE")
        print(" Install CuPy for GPU acceleration")
        print("=" * 70)
    
    window = MainWindow()
    window.show()
    
    app.exec()
    
    # Cleanup
    if GPU_AVAILABLE:
        clear_gpu_cache()