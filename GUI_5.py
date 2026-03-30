import os
import numpy as np
import pydicom
import cv2
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QTime, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter, QToolBar,
    QComboBox, QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMenu, QTabWidget
)
from skimage.measure import shannon_entropy
from math import log10

try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False
    print("Warning: CuPy not available. GPU acceleration disabled.")

# =========================================================
# Image Processing Functions
# =========================================================

def to_float(img, bit_depth=16):
    """Convert image to float32 in range [0, 1]"""
    max_val = float(2**bit_depth - 1)
    return img.astype(np.float32) / max_val

def from_float(img_float, bit_depth=16):
    """Convert float32 [0,1] image to uint16"""
    max_val = float(2**bit_depth - 1)
    return np.clip(img_float * max_val, 0, max_val).astype(np.uint16)

def gamma_correction(img_float, gamma):
    return np.clip(np.power(img_float, gamma), 0, 1)

def apply_clahe(img_float, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_uint16 = np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_uint16)
    return np.clip(clahe_img.astype(np.float32) / 65535, 0, 1)

def wiener_filter(img_float, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    local_mean = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_mean_sq = cv2.filter2D(img_float**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-10)
    noise_var = np.mean(local_var) * 0.01
    gain = np.clip((local_var - noise_var) / local_var, 0, 1)
    result = local_mean + gain * (img_float - local_mean)
    return np.clip(result, 0, 1)

def apply_windowing(img_float, WL=0.5, WW=0.7):
    img_min = WL - WW / 2
    img_max = WL + WW / 2
    windowed = np.clip(img_float, img_min, img_max)
    return np.clip((windowed - img_min) / (img_max - img_min), 0, 1)

def apply_unsharp_mask(img_float, kernel_size=(5,5), sigma=1.0, amount=1.5):
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    sharpened = (amount + 1) * img_float - amount * blurred
    return np.clip(sharpened, 0, 1)

def gaussian_denoise(img_float, sigma=0.6):
    return cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)

def nlm_denoise(img_float, h=0.0125):
    img_8u = np.uint8(np.clip(img_float * 255, 0, 255))
    denoised = cv2.fastNlMeansDenoising(img_8u, None, h=h*255, templateWindowSize=7, searchWindowSize=21)
    return denoised.astype(np.float32) / 255.0

def process_frame(frame, gamma, clahe_clip=2.0, clahe_tiles=(8, 8), wiener_kernel=5):
    """Complete preprocessing pipeline"""
    frame_float = to_float(frame)
    wi = wiener_filter(frame_float, kernel_size=wiener_kernel)
    gm = gamma_correction(wi, gamma=gamma)
    cl = apply_clahe(gm, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    unsharp = apply_unsharp_mask(cl)
    win = apply_windowing(unsharp)
    return win

# =========================================================
# GPU Phase Correlation Functions
# =========================================================

DICOM_PIXEL_MAX = 65535.0

def _prep_for_fft(img_gpu: cp.ndarray) -> cp.ndarray:
    """Prepare image for FFT: apply Hann window and zero-mean."""
    img = img_gpu.astype(cp.float32, copy=False)
    h, w = img.shape
    
    hann_y = cp.hanning(h).reshape(h, 1)
    hann_x = cp.hanning(w).reshape(1, w)
    window = hann_y * hann_x
    
    windowed = img * window
    return windowed - windowed.mean()


def _parabolic_offset_1d(fm1: float, f0: float, fp1: float) -> float:
    """1D parabola fitting for subpixel accuracy."""
    denom = (fm1 - 2.0 * f0 + fp1)
    if abs(denom) < 1e-10:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom


def phase_correlation_gpu(mask_gpu: cp.ndarray, live_gpu: cp.ndarray):
    """
    Compute phase correlation shift between two 16-bit images on GPU.
    Returns dy, dx (signed pixel shifts) and correlation surface.
    """
    a = _prep_for_fft(mask_gpu)
    b = _prep_for_fft(live_gpu)

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

    return float(dy), float(dx), corr


def _bilinear_shift_gpu(img_gpu: cp.ndarray, dy: float, dx: float) -> cp.ndarray:
    """Apply bilinear interpolation shift on GPU."""
    h, w = img_gpu.shape
    
    y_coords = cp.arange(h, dtype=cp.float32)
    x_coords = cp.arange(w, dtype=cp.float32)
    yy, xx = cp.meshgrid(y_coords, x_coords, indexing='ij')
    
    y_src = yy - dy
    x_src = xx - dx
    
    y_src = cp.clip(y_src, 0, h - 1)
    x_src = cp.clip(x_src, 0, w - 1)
    
    y_int = cp.floor(y_src).astype(cp.int32)
    x_int = cp.floor(x_src).astype(cp.int32)
    y_frac = y_src - y_int
    x_frac = x_src - x_int
    
    y_int = cp.clip(y_int, 0, h - 1)
    x_int = cp.clip(x_int, 0, w - 1)
    y_int_next = cp.clip(y_int + 1, 0, h - 1)
    x_int_next = cp.clip(x_int + 1, 0, w - 1)
    
    v00 = img_gpu[y_int, x_int]
    v01 = img_gpu[y_int, x_int_next]
    v10 = img_gpu[y_int_next, x_int]
    v11 = img_gpu[y_int_next, x_int_next]
    
    v0 = v00 * (1 - x_frac) + v01 * x_frac
    v1 = v10 * (1 - x_frac) + v11 * x_frac
    shifted = v0 * (1 - y_frac) + v1 * y_frac
    
    return cp.clip(shifted, 0, DICOM_PIXEL_MAX).astype(cp.float32, copy=False)


def apply_subpixel_shift_gpu(img_gpu: cp.ndarray, dy: float, dx: float) -> cp.ndarray:
    """Apply subpixel shift using appropriate method based on shift magnitude."""
    SHIFT_THRESHOLD = 0.1
    
    if abs(dy) < SHIFT_THRESHOLD and abs(dx) < SHIFT_THRESHOLD:
        return img_gpu.astype(cp.float32, copy=False)
    
    h, w = img_gpu.shape
    img_float = img_gpu.astype(cp.float32, copy=False)
    
    if abs(dy) < 0.5 and abs(dx) < 0.5:
        return _bilinear_shift_gpu(img_float, dy, dx)
    
    F = cp.fft.fftn(img_float)
    
    ky = cp.fft.fftfreq(h).reshape(-1, 1)
    kx = cp.fft.fftfreq(w).reshape(1, -1)
    
    phase = cp.exp(-2j * cp.pi * (ky * dy + kx * dx))
    shifted = cp.fft.ifftn(F * phase).real
    
    return cp.clip(shifted, 0, DICOM_PIXEL_MAX).astype(cp.float32, copy=False)


def histogram_matching_gpu(source_gpu: cp.ndarray, reference_gpu: cp.ndarray, bins: int = 256) -> cp.ndarray:
    """Match histogram of source to reference on GPU."""
    source = source_gpu.astype(cp.float32, copy=False)
    reference = reference_gpu.astype(cp.float32, copy=False)
    
    src_hist, src_bins = cp.histogram(source.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))
    ref_hist, ref_bins = cp.histogram(reference.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))
    
    src_hist = src_hist.astype(cp.float32) + 1e-6
    ref_hist = ref_hist.astype(cp.float32) + 1e-6
    
    src_cdf = cp.cumsum(src_hist)
    ref_cdf = cp.cumsum(ref_hist)
    
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    src_centers = src_bins[:-1] + (src_bins[1] - src_bins[0]) / 2.0
    ref_centers = ref_bins[:-1] + (ref_bins[1] - ref_bins[0]) / 2.0
    
    lookup_table = cp.interp(src_cdf, ref_cdf, ref_centers)
    
    indices = cp.searchsorted(src_centers, source.ravel(), side='left')
    indices = cp.clip(indices, 0, len(lookup_table) - 1)
    
    matched = lookup_table[indices].reshape(source.shape)
    
    return cp.clip(matched, 0, DICOM_PIXEL_MAX).astype(cp.float32, copy=False)


def estimate_affine_transform_gpu(src_img, dst_img, use_gpu=True):
    """Estimate affine transformation using phase correlation on GPU."""
    if use_gpu and GPU_ENABLED:
        try:
            src_gpu = cp.asarray(src_img.astype(np.float32))
            dst_gpu = cp.asarray(dst_img.astype(np.float32))
            
            dy, dx, _ = phase_correlation_gpu(dst_gpu, src_gpu)
            
            return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        except Exception as e:
            print(f"GPU phase correlation failed: {e}. Returning identity.")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    else:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

def apply_affine_transform(img_float, transform_matrix):
    """Apply affine transformation to image"""
    h, w = img_float.shape
    transform_matrix = transform_matrix.astype(np.float32)
    registered = cv2.warpAffine(img_float, transform_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
    return np.clip(registered, 0, 1)

def perform_dsa(mask_frame, contrast_frame, normalization='standard'):
    """Perform Digital Subtraction Angiography"""
    if normalization == 'standard':
        dsa = contrast_frame - mask_frame
    elif normalization == 'weighted':
        epsilon = 1e-6
        dsa = (contrast_frame - mask_frame) / (mask_frame + epsilon)
    elif normalization == 'log':
        epsilon = 1e-6
        dsa = np.log(mask_frame + epsilon) - np.log(contrast_frame + epsilon)
    else:
        dsa = contrast_frame - mask_frame
    
    dsa_min, dsa_max = dsa.min(), dsa.max()
    if dsa_max > dsa_min:
        dsa_normalized = (dsa - dsa_min) / (dsa_max - dsa_min)
    else:
        dsa_normalized = np.zeros_like(dsa)
    
    return np.clip(dsa_normalized, 0, 1)

# =========================================================
# Metrics Calculation
# =========================================================

def calc_snr(roi):
    return np.mean(roi) / (np.std(roi) + 1e-8)

def calc_cnr(signal_roi, background_roi):
    return abs(np.mean(signal_roi) - np.mean(background_roi)) / (np.std(background_roi) + 1e-8)

def calc_entropy(roi):
    return shannon_entropy(roi)

def extract_rois_from_frame(frame, rois):
    return [frame[y1:y2, x1:x2] for (x1, x2, y1, y2) in rois]

# =========================================================
# Processing Threads
# =========================================================

class DSAProcessingThread(QThread):
    """Background thread for DSA processing with GPU phase correlation"""
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
                    wiener_kernel=self.params['wiener_kernel']
                )
                frames_preprocessed.append(processed)
                progress_pct = 10 + int(30 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Preprocessing frame {i+1}/{len(self.frames)}")
            
            # Step 2: Get mask frame
            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_preprocessed[mask_idx]
            self.progress.emit(45, f"Using frame {mask_idx} as mask")
            
            # Load mask to GPU if available
            if GPU_ENABLED:
                mask_gpu = cp.asarray(mask_frame.astype(np.float32))
                self.progress.emit(48, "Mask frame loaded to GPU")
            
            # Step 3: Register frames using GPU phase correlation
            self.progress.emit(50, "Registering frames with GPU phase correlation...")
            for i, frame in enumerate(frames_preprocessed):
                if i == mask_idx:
                    frames_registered.append(frame)
                else:
                    if GPU_ENABLED:
                        try:
                            frame_gpu = cp.asarray(frame.astype(np.float32))
                            dy, dx, _ = phase_correlation_gpu(mask_gpu, frame_gpu)
                            registered_gpu = apply_subpixel_shift_gpu(frame_gpu, dy, dx)
                            registered = cp.asnumpy(registered_gpu)
                        except Exception as e:
                            self.progress.emit(50, f"GPU registration failed, using CPU: {str(e)[:30]}")
                            transform = estimate_affine_transform_gpu(frame, mask_frame, use_gpu=False)
                            registered = apply_affine_transform(frame, transform)
                    else:
                        transform = estimate_affine_transform_gpu(frame, mask_frame, use_gpu=False)
                        registered = apply_affine_transform(frame, transform)
                    
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
                        mask_frame, contrast_frame,
                        normalization=self.params['dsa_normalization']
                    )
                frames_dsa.append(dsa_result)
                
                progress_pct = 85 + int(15 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"DSA frame {i+1}/{len(self.frames)}")
            
            self.progress.emit(100, "Processing complete!")
            self.finished.emit(frames_preprocessed, frames_registered, frames_dsa)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress.emit(0, f"Error: {str(e)}")

class PostProcessingThread(QThread):
    """Background thread for post-processing with metrics"""
    progress = Signal(int, str)
    finished = Signal(list, pd.DataFrame)
    
    def __init__(self, frames, params, rois):
        super().__init__()
        self.frames = frames
        self.params = params
        self.rois = rois
        
    def run(self):
        try:
            processed_frames = []
            metrics = []
            
            self.progress.emit(5, "Starting post-processing pipeline...")
            
            for i, frame in enumerate(self.frames):
                img = to_float(frame)
                
                # Apply post-processing pipeline
                img = gamma_correction(img, gamma=self.params['gamma'])
                img = gaussian_denoise(img, sigma=self.params['gaussian_sigma'])
                img = nlm_denoise(img, h=self.params['nlm_h'])
                img = clahe_enhance(img, clip_limit=self.params['clahe_clip'], 
                                   tile_grid_size=self.params['clahe_tiles'])
                img = apply_unsharp_mask_pp(img, amount=self.params['unsharp_amount'], 
                                           radius=self.params['unsharp_radius'])
                
                proc_uint16 = from_float(img)
                processed_frames.append(proc_uint16)
                
                # Calculate metrics
                orig_f = to_float(frame)
                proc_f = to_float(proc_uint16)
                orig_rois = extract_rois_from_frame(orig_f, self.rois)
                proc_rois = extract_rois_from_frame(proc_f, self.rois)
                
                for j, (orig_roi, proc_roi) in enumerate(zip(orig_rois, proc_rois)):
                    snr_orig = calc_snr(orig_roi)
                    snr_proc = calc_snr(proc_roi)
                    cnr_orig = calc_cnr(orig_roi, orig_rois[0]) if j != 0 else np.nan
                    cnr_proc = calc_cnr(proc_roi, proc_rois[0]) if j != 0 else np.nan
                    ent_orig = calc_entropy(orig_roi)
                    ent_proc = calc_entropy(proc_roi)
                    
                    metrics.append({
                        "Frame": i,
                        "ROI": j,
                        "SNR_Original": snr_orig,
                        "SNR_Processed": snr_proc,
                        "CNR_Original": cnr_orig,
                        "CNR_Processed": cnr_proc,
                        "Entropy_Original": ent_orig,
                        "Entropy_Processed": ent_proc
                    })
                
                progress_pct = 10 + int(90 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Processing frame {i+1}/{len(self.frames)}")
            
            self.progress.emit(100, "Post-processing complete!")
            df = pd.DataFrame(metrics)
            self.finished.emit(processed_frames, df)
            
        except Exception as e:
            self.progress.emit(0, f"Error: {str(e)}")

def clahe_enhance(img_float, clip_limit=2, tile_grid_size=(4,4)):
    img_8u = np.uint8(img_float * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_8u)
    return enhanced.astype(np.float32) / 255.0

def apply_unsharp_mask_pp(img_float, amount=0.45, radius=0.35):
    blurred = cv2.GaussianBlur(img_float, (0,0), radius)
    sharpened = cv2.addWeighted(img_float, 1+amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)

# =========================================================
# DICOM Viewer Widget
# =========================================================

class DicomViewer(QWidget):
    """DICOM viewer with advanced DSA processing"""
    def __init__(self, parent=None, dsa_mode=False):
        super().__init__(parent)
        self.frames = None
        self.frame_index = 0
        self.dsa_mode = dsa_mode
        
        # Processing parameters
        self.gamma = 0.5
        self.clahe_clip = 2.0
        self.clahe_tiles = (8, 8)
        self.wiener_kernel = 5
        self.mask_frame_index = 0
        self.registration_method = 'ecc'
        self.dsa_normalization = 'standard'
        
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
        title = "DSA Viewer (Advanced Pipeline)" if dsa_mode else "Original DICOM"
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
            self.save_btn = QPushButton("💾 Save DSA Results")
            self.save_btn.clicked.connect(self.show_save_menu)
            self.save_btn.setEnabled(False)
            layout.addWidget(self.save_btn)

    def setup_dsa_controls(self, layout):
        """Setup DSA processing controls"""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Row 1: Preprocessing parameters
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Gamma:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 2.0)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.setSingleStep(0.1)
        row1.addWidget(self.gamma_spin)
        
        row1.addWidget(QLabel("CLAHE Clip:"))
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(1.0, 5.0)
        self.clahe_spin.setValue(2.0)
        self.clahe_spin.setSingleStep(0.5)
        row1.addWidget(self.clahe_spin)
        
        row1.addWidget(QLabel("Wiener Kernel:"))
        self.wiener_spin = QSpinBox()
        self.wiener_spin.setRange(3, 11)
        self.wiener_spin.setValue(5)
        self.wiener_spin.setSingleStep(2)
        row1.addWidget(self.wiener_spin)
        
        controls_layout.addLayout(row1)
        
        # Row 2: Registration parameters
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row2.addWidget(self.mask_spin)
        
        row2.addWidget(QLabel("GPU Acceleration:"))
        self.gpu_check = QCheckBox()
        self.gpu_check.setChecked(GPU_ENABLED)
        self.gpu_check.setEnabled(GPU_ENABLED)
        row2.addWidget(self.gpu_check)
        
        controls_layout.addLayout(row2)
        
        # Row 3: DSA normalization
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("DSA Method:"))
        self.dsa_combo = QComboBox()
        self.dsa_combo.addItems(['Standard', 'Weighted', 'Logarithmic'])
        row3.addWidget(self.dsa_combo)
        
        self.process_btn = QPushButton("🔄 Process DSA Pipeline")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        row3.addWidget(self.process_btn)
        
        controls_layout.addLayout(row3)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_widget)

    def show_save_menu(self):
        """Show save options menu"""
        menu = QMenu(self)
        
        menu.addAction("Save Current Frame as PNG", lambda: self.save_current_frame('png'))
        menu.addAction("Save Current Frame as TIFF (16-bit)", lambda: self.save_current_frame('tiff'))
        menu.addSeparator()
        menu.addAction("Save DSA Result as DICOM (.dcm)", self.save_dsa_as_dicom)
        menu.addSeparator()
        menu.addAction("Save All DSA Frames as PNG Sequence", lambda: self.save_sequence('png'))
        menu.addAction("Save All DSA Frames as TIFF Sequence", lambda: self.save_sequence('tiff'))
        menu.addSeparator()
        menu.addAction("Export DSA as Video (MP4)", self.save_video)
        menu.addAction("Export DSA as AVI", self.save_avi)
        menu.addSeparator()
        menu.addAction("Save Preprocessed Frames", self.save_preprocessed)
        menu.addAction("Save Registered Frames", self.save_registered)
        
        menu.exec_(self.save_btn.mapToGlobal(self.save_btn.rect().bottomLeft()))

    def save_current_frame(self, format_type):
        """Save current DSA frame"""
        if self.frames_dsa is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return
        
        ext = 'png' if format_type == 'png' else 'tif'
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save Current Frame",
            f"dsa_frame_{self.frame_index:03d}.{ext}",
            f"{ext.upper()} Files (*.{ext})"
        )
        
        if filename:
            try:
                frame = self.frames_dsa[self.frame_index]
                if format_type == 'png':
                    img_uint8 = (frame * 255).astype(np.uint8)
                    cv2.imwrite(filename, img_uint8)
                else:  # tiff
                    img_uint16 = (frame * 65535).astype(np.uint16)
                    cv2.imwrite(filename, img_uint16)
                
                QMessageBox.information(self, "Success", f"Frame saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save frame:\n{str(e)}")

    def save_sequence(self, format_type):
        """Save all DSA frames as sequence"""
        if self.frames_dsa is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return
        
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        
        try:
            ext = 'png' if format_type == 'png' else 'tif'
            for i, frame in enumerate(self.frames_dsa):
                filename = os.path.join(folder, f"dsa_frame_{i:03d}.{ext}")
                if format_type == 'png':
                    img_uint8 = (frame * 255).astype(np.uint8)
                    cv2.imwrite(filename, img_uint8)
                else:  # tiff
                    img_uint16 = (frame * 65535).astype(np.uint16)
                    cv2.imwrite(filename, img_uint16)
            
            QMessageBox.information(self, "Success", 
                f"Saved {len(self.frames_dsa)} frames to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save sequence:\n{str(e)}")

    def save_video(self):
        """Export DSA as MP4 video"""
        self._save_video_format('mp4', 'mp4v')

    def save_avi(self):
        """Export DSA as AVI video"""
        self._save_video_format('avi', 'XVID')

    def _save_video_format(self, ext, fourcc_code):
        """Save video in specified format"""
        if self.frames_dsa is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save as {ext.upper()}",
            f"dsa_output.{ext}",
            f"{ext.upper()} Files (*.{ext})"
        )
        
        if filename:
            try:
                h, w = self.frames_dsa[0].shape
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                fps = 10  # Default FPS
                out = cv2.VideoWriter(filename, fourcc, fps, (w, h), False)
                
                for frame in self.frames_dsa:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    out.write(frame_uint8)
                
                out.release()
                QMessageBox.information(self, "Success", f"Video saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save video:\n{str(e)}")

        def save_dsa_as_dicom(self):
            """Save DSA result as DICOM file"""
        if self.frames_dsa is None:
            QMessageBox.warning(self, "No Data", "Please process DSA first!")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for DICOM Export")
        if not folder:
            return

        try:
            # Try to get metadata from the original file if loaded
            ds_template = None
            if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'dicom_dataset'):
                ds_template = self.parent().parent().dicom_dataset

            for i, frame in enumerate(self.frames_dsa):
                # Convert to 16-bit
                img_uint16 = (frame * 65535).astype(np.uint16)

                # Create a new DICOM dataset
                ds = pydicom.Dataset()
                ds.file_meta = pydicom.Dataset()
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                ds.is_little_endian = True
                ds.is_implicit_VR = True

                # Basic DICOM attributes
                ds.PatientName = "DSA_Patient"
                ds.PatientID = "0000"
                ds.Modality = "XA"
                ds.SeriesDescription = "DSA Processed Result"
                ds.InstanceNumber = i + 1
                ds.Rows, ds.Columns = img_uint16.shape
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.PixelData = img_uint16.tobytes()

                # Optional: Copy metadata from source file
                if ds_template is not None:
                    for elem in ds_template:
                        if elem.tag not in ds:
                            ds.add(elem)

                filename = os.path.join(folder, f"dsa_frame_{i:03d}.dcm")
                pydicom.dcmwrite(filename, ds)

            QMessageBox.information(self, "Success", f"Saved {len(self.frames_dsa)} DSA frames as DICOMs in:\n{folder}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save DICOM files:\n{str(e)}")


    def save_preprocessed(self):
        """Save preprocessed frames"""
        self._save_processed_frames(self.frames_preprocessed, "preprocessed")

    def save_registered(self):
        """Save registered frames"""
        self._save_processed_frames(self.frames_registered, "registered")

    def _save_processed_frames(self, frames, label):
        """Save processed frames as sequence"""
        if frames is None:
            QMessageBox.warning(self, "No Data", f"No {label} frames available!")
            return
        
        folder = QFileDialog.getExistingDirectory(self, f"Select Output Folder for {label.title()} Frames")
        if not folder:
            return
        
        try:
            for i, frame in enumerate(frames):
                filename = os.path.join(folder, f"{label}_frame_{i:03d}.tif")
                img_uint16 = (frame * 65535).astype(np.uint16)
                cv2.imwrite(filename, img_uint16)
            
            QMessageBox.information(self, "Success", 
                f"Saved {len(frames)} {label} frames to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save {label} frames:\n{str(e)}")

    def set_frames(self, frames, filename=""):
        """Set frames for viewing"""
        self.frames = frames
        self.frame_index = 0
        
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
        params = {
            'gamma': self.gamma_spin.value(),
            'clahe_clip': self.clahe_spin.value(),
            'clahe_tiles': (8, 8),
            'wiener_kernel': self.wiener_spin.value(),
            'mask_frame_index': self.mask_spin.value(),
            'use_gpu': self.gpu_check.isChecked() and GPU_ENABLED,
            'dsa_normalization': self.dsa_combo.currentText().lower()
        }
        
        # Update stored parameters
        self.gamma = params['gamma']
        self.clahe_clip = params['clahe_clip']
        self.wiener_kernel = params['wiener_kernel']
        self.mask_frame_index = params['mask_frame_index']
        self.dsa_normalization = params['dsa_normalization']
        
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
# Post-Processing Viewer Widget
# =========================================================

class PostProcessingViewer(QWidget):
    """Post-processing viewer with metrics calculation"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = None
        self.frame_index = 0
        self.frames_postprocessed = None
        self.metrics_df = None
        self.processing_thread = None
        
        # Post-processing parameters
        self.gamma = 0.95
        self.gaussian_sigma = 0.6
        self.nlm_h = 0.0125
        self.clahe_clip = 2
        self.clahe_tiles = (4, 4)
        self.unsharp_amount = 0.45
        self.unsharp_radius = 0.35
        
        # Default ROIs
        self.rois = [
            (300, 350, 200, 250),
            (515, 565, 725, 775),
            (640, 690, 440, 490),
            (825, 875, 630, 680)
        ]
        
        layout = QVBoxLayout(self)
        
        # File info
        self.filename_label = QLabel("No file loaded")
        self.info_label = QLabel("")
        layout.addWidget(self.filename_label)
        layout.addWidget(self.info_label)
        
        # Image display
        self.image_label = QLabel("Post-Processing Viewer")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label, 1)
        
        # Post-processing controls
        self.setup_postprocessing_controls(layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Save button
        self.save_btn = QPushButton("💾 Save Post-Processed Results")
        self.save_btn.clicked.connect(self.show_save_menu)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        # Metrics button
        self.metrics_btn = QPushButton("📊 Export Metrics to Excel")
        self.metrics_btn.clicked.connect(self.export_metrics)
        self.metrics_btn.setEnabled(False)
        layout.addWidget(self.metrics_btn)

    def setup_postprocessing_controls(self, layout):
        """Setup post-processing controls"""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Row 1: Gamma and Denoising
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Gamma:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 2.0)
        self.gamma_spin.setValue(0.95)
        self.gamma_spin.setSingleStep(0.05)
        row1.addWidget(self.gamma_spin)
        
        row1.addWidget(QLabel("Gaussian Sigma:"))
        self.gaussian_spin = QDoubleSpinBox()
        self.gaussian_spin.setRange(0.1, 2.0)
        self.gaussian_spin.setValue(0.6)
        self.gaussian_spin.setSingleStep(0.1)
        row1.addWidget(self.gaussian_spin)
        
        row1.addWidget(QLabel("NLM h:"))
        self.nlm_spin = QDoubleSpinBox()
        self.nlm_spin.setRange(0.001, 0.1)
        self.nlm_spin.setValue(0.0125)
        self.nlm_spin.setSingleStep(0.001)
        row1.addWidget(self.nlm_spin)
        
        controls_layout.addLayout(row1)
        
        # Row 2: CLAHE and Unsharp Mask
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("CLAHE Clip:"))
        self.clahe_spin = QSpinBox()
        self.clahe_spin.setRange(1, 10)
        self.clahe_spin.setValue(2)
        row2.addWidget(self.clahe_spin)
        
        row2.addWidget(QLabel("Unsharp Amount:"))
        self.unsharp_amount_spin = QDoubleSpinBox()
        self.unsharp_amount_spin.setRange(0.0, 2.0)
        self.unsharp_amount_spin.setValue(0.45)
        self.unsharp_amount_spin.setSingleStep(0.05)
        row2.addWidget(self.unsharp_amount_spin)
        
        row2.addWidget(QLabel("Unsharp Radius:"))
        self.unsharp_radius_spin = QDoubleSpinBox()
        self.unsharp_radius_spin.setRange(0.1, 2.0)
        self.unsharp_radius_spin.setValue(0.35)
        self.unsharp_radius_spin.setSingleStep(0.05)
        row2.addWidget(self.unsharp_radius_spin)
        
        controls_layout.addLayout(row2)
        
        # Row 3: Process button
        row3 = QHBoxLayout()
        self.process_btn = QPushButton("🔄 Process & Calculate Metrics")
        self.process_btn.clicked.connect(self.process_postprocessing)
        row3.addWidget(self.process_btn)
        
        controls_layout.addLayout(row3)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_widget)

    def set_frames(self, frames, filename=""):
        """Set frames for post-processing"""
        self.frames = frames
        self.frame_index = 0
        self.frames_postprocessed = None
        self.metrics_df = None
        
        if filename:
            self.filename_label.setText(filename)
        self.info_label.setText(f"{len(frames)} frames")
        self.save_btn.setEnabled(False)
        self.metrics_btn.setEnabled(False)
        self.show_frame()

    def process_postprocessing(self):
        """Start post-processing in background thread"""
        if self.frames is None:
            return
        
        params = {
            'gamma': self.gamma_spin.value(),
            'gaussian_sigma': self.gaussian_spin.value(),
            'nlm_h': self.nlm_spin.value(),
            'clahe_clip': self.clahe_spin.value(),
            'clahe_tiles': (4, 4),
            'unsharp_amount': self.unsharp_amount_spin.value(),
            'unsharp_radius': self.unsharp_radius_spin.value()
        }
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Status: Processing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        self.processing_thread = PostProcessingThread(self.frames, params, self.rois)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Status: {message}")

    def on_processing_finished(self, processed_frames, metrics_df):
        """Handle processing completion"""
        self.frames_postprocessed = processed_frames
        self.metrics_df = metrics_df
        
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.metrics_btn.setEnabled(True)
        self.status_label.setText("Status: Processing Complete!")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.show_frame()

    def show_frame(self):
        """Display current frame"""
        if self.frames is None:
            return
        
        try:
            if self.frames_postprocessed is not None:
                frame = self.frames_postprocessed[self.frame_index]
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

    def set_frame_index(self, idx):
        """Set frame index"""
        if self.frames is None:
            return
        self.frame_index = idx % len(self.frames)
        self.show_frame()

    def show_save_menu(self):
        """Show save options menu"""
        menu = QMenu(self)
        menu.addAction("Save All Frames as TIFF Sequence", lambda: self.save_sequence('tiff'))
        menu.addAction("Save All Frames as PNG Sequence", lambda: self.save_sequence('png'))
        menu.exec_(self.save_btn.mapToGlobal(self.save_btn.rect().bottomLeft()))

    def save_sequence(self, format_type):
        """Save post-processed frames"""
        if self.frames_postprocessed is None:
            QMessageBox.warning(self, "No Data", "Please process frames first!")
            return
        
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        
        try:
            ext = 'png' if format_type == 'png' else 'tif'
            for i, frame in enumerate(self.frames_postprocessed):
                filename = os.path.join(folder, f"postprocessed_frame_{i:03d}.{ext}")
                if format_type == 'png':
                    img_uint8 = (frame * 255).astype(np.uint8)
                    cv2.imwrite(filename, img_uint8)
                else:
                    img_uint16 = (frame * 65535).astype(np.uint16)
                    cv2.imwrite(filename, img_uint16)
            
            QMessageBox.information(self, "Success", 
                f"Saved {len(self.frames_postprocessed)} frames to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save frames:\n{str(e)}")

    def export_metrics(self):
        """Export metrics to Excel"""
        if self.metrics_df is None:
            QMessageBox.warning(self, "No Data", "Please process frames first!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics Excel",
            "metrics.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if filename:
            try:
                self.metrics_df.to_excel(filename, index=False)
                QMessageBox.information(self, "Success", f"Metrics saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save metrics:\n{str(e)}")

# =========================================================
# Main Window
# =========================================================

class Desktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced DSA Viewer - Full Pipeline Integration with Post-Processing")
        self.resize(1800, 900)
        
        # Enable drag and drop
        self.setAcceptDrops(True)

        # Main splitter with tabs
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Tab widget for different viewers
        self.tabs = QTabWidget()
        
        # Original viewer
        self.viewer1 = DicomViewer(dsa_mode=False)
        self.tabs.addTab(self.viewer1, "Original DICOM")
        
        # DSA viewer
        self.viewer2 = DicomViewer(dsa_mode=True)
        self.tabs.addTab(self.viewer2, "DSA Processing")
        
        # Post-processing viewer
        self.viewer3 = PostProcessingViewer()
        self.tabs.addTab(self.viewer3, "Post-Processing & Metrics")
        
        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

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
        self.pause_btn = QPushButton("⸸ Pause")
        self.stop_btn = QPushButton("⹿ Stop")
        
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
            
            # Update all viewers
            short_name = os.path.basename(filename)
            self.viewer1.set_frames(frames, short_name)
            self.viewer2.set_frames(frames, short_name)
            self.viewer3.set_frames(frames, short_name)
            
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
        """Update all viewers"""
        self.viewer1.set_frame_index(self.frame_index)
        self.viewer2.set_frame_index(self.frame_index)
        self.viewer3.set_frame_index(self.frame_index)

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
    
    window = Desktop()
    window.show()
    
    app.exec()