import os
import numpy as np
import pydicom
import cv2
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter, QToolBar,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, 
    QCheckBox, QMenu
)
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy (GPU) available")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not available - using CPU only")

try:
    import cv2.cuda as cv2cuda
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        print(f"✓ OpenCV CUDA available - {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
except:
    CUDA_AVAILABLE = False
    print("✗ OpenCV CUDA not available")

# =========================================================
# GPU-Accelerated Image Processing Functions
# =========================================================

class GPUProcessor:
    """GPU-accelerated image processing using CuPy"""
    
    @staticmethod
    def to_float(img, bit_depth=16):
        if GPU_AVAILABLE:
            img_gpu = cp.asarray(img, dtype=cp.float32)
            max_val = float(2**bit_depth - 1)
            return img_gpu / max_val
        else:
            max_val = float(2**bit_depth - 1)
            return img.astype(np.float32) / max_val
    
    @staticmethod
    def from_float(img_float, bit_depth=16):
        max_val = float(2**bit_depth - 1)
        if GPU_AVAILABLE:
            return cp.clip(img_float * max_val, 0, max_val).astype(cp.uint16)
        else:
            return np.clip(img_float * max_val, 0, max_val).astype(np.uint16)
    
    @staticmethod
    def gamma_correction(img_float, gamma):
        if GPU_AVAILABLE:
            return cp.clip(cp.power(img_float, gamma), 0, 1)
        else:
            return np.clip(np.power(img_float, gamma), 0, 1)
    
    @staticmethod
    def apply_clahe_gpu(img_float, clip_limit=2.0, tile_grid_size=(8, 8)):
        """CLAHE with GPU support"""
        if CUDA_AVAILABLE:
            # Use OpenCV CUDA CLAHE
            img_uint16 = np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
            gpu_img = cv2cuda.GpuMat()
            gpu_img.upload(img_uint16)
            
            clahe = cv2cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            gpu_result = clahe.apply(gpu_img, cv2cuda.Stream_Null())
            
            result = gpu_result.download()
            return np.clip(result.astype(np.float32) / 65535, 0, 1)
        else:
            # CPU fallback
            img_uint16 = np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            clahe_img = clahe.apply(img_uint16)
            return np.clip(clahe_img.astype(np.float32) / 65535, 0, 1)
    
    @staticmethod
    def wiener_filter_gpu(img_float, kernel_size=5):
        """GPU-accelerated Wiener filter"""
        if GPU_AVAILABLE:
            kernel = cp.ones((kernel_size, kernel_size), cp.float32) / (kernel_size**2)
            
            # Convert to CuPy if not already
            if isinstance(img_float, np.ndarray):
                img_float = cp.asarray(img_float)
            
            # Use CuPy's signal processing
            from cupyx.scipy.ndimage import convolve
            local_mean = convolve(img_float, kernel, mode='reflect')
            local_mean_sq = convolve(img_float**2, kernel, mode='reflect')
            
            local_var = cp.maximum(local_mean_sq - local_mean**2, 1e-10)
            noise_var = cp.mean(local_var) * 0.01
            gain = cp.clip((local_var - noise_var) / local_var, 0, 1)
            result = local_mean + gain * (img_float - local_mean)
            
            return cp.clip(result, 0, 1)
        else:
            # CPU fallback
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            local_mean = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)
            local_mean_sq = cv2.filter2D(img_float**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
            local_var = np.maximum(local_mean_sq - local_mean**2, 1e-10)
            noise_var = np.mean(local_var) * 0.01
            gain = np.clip((local_var - noise_var) / local_var, 0, 1)
            result = local_mean + gain * (img_float - local_mean)
            return np.clip(result, 0, 1)
    
    @staticmethod
    def apply_unsharp_mask_gpu(img_float, kernel_size=(5,5), sigma=1.0, amount=1.5):
        """GPU-accelerated unsharp mask"""
        if CUDA_AVAILABLE:
            gpu_img = cv2cuda.GpuMat()
            if isinstance(img_float, np.ndarray):
                gpu_img.upload(img_float.astype(np.float32))
            else:
                gpu_img.upload(cp.asnumpy(img_float).astype(np.float32))
            
            gaussian_filter = cv2cuda.createGaussianFilter(
                cv2.CV_32F, cv2.CV_32F, kernel_size, sigma
            )
            gpu_blurred = gaussian_filter.apply(gpu_img)
            
            blurred = gpu_blurred.download()
            img_cpu = gpu_img.download()
            
            sharpened = (amount + 1) * img_cpu - amount * blurred
            return np.clip(sharpened, 0, 1)
        elif GPU_AVAILABLE:
            if isinstance(img_float, np.ndarray):
                img_float = cp.asarray(img_float)
            
            from cupyx.scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(img_float, sigma)
            sharpened = (amount + 1) * img_float - amount * blurred
            return cp.clip(sharpened, 0, 1)
        else:
            blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
            sharpened = (amount + 1) * img_float - amount * blurred
            return np.clip(sharpened, 0, 1)
    
    @staticmethod
    def apply_windowing_gpu(img_float, WL=0.5, WW=0.7):
        """GPU-accelerated windowing"""
        img_min = WL - WW / 2
        img_max = WL + WW / 2
        
        if GPU_AVAILABLE and isinstance(img_float, cp.ndarray):
            windowed = cp.clip(img_float, img_min, img_max)
            return cp.clip((windowed - img_min) / (img_max - img_min), 0, 1)
        else:
            windowed = np.clip(img_float, img_min, img_max)
            return np.clip((windowed - img_min) / (img_max - img_min), 0, 1)

def process_frame_gpu(frame, gamma, clahe_clip=2.0, clahe_tiles=(8, 8), wiener_kernel=5, use_gpu=True):
    """GPU-accelerated preprocessing pipeline"""
    gpu_proc = GPUProcessor()
    
    try:
        if use_gpu and GPU_AVAILABLE:
            # Process on GPU
            frame_float = gpu_proc.to_float(frame)
            wi = gpu_proc.wiener_filter_gpu(frame_float, kernel_size=wiener_kernel)
            gm = gpu_proc.gamma_correction(wi, gamma=gamma)
            
            # Convert to CPU for CLAHE (more efficient)
            gm_cpu = cp.asnumpy(gm) if isinstance(gm, cp.ndarray) else gm
            cl = gpu_proc.apply_clahe_gpu(gm_cpu, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
            
            # Back to GPU for final steps
            if GPU_AVAILABLE:
                cl = cp.asarray(cl)
            
            unsharp = gpu_proc.apply_unsharp_mask_gpu(cl)
            win = gpu_proc.apply_windowing_gpu(unsharp)
            
            # Return as numpy array
            return cp.asnumpy(win) if isinstance(win, cp.ndarray) else win
        else:
            # CPU fallback
            frame_float = gpu_proc.to_float(frame)
            wi = gpu_proc.wiener_filter_gpu(frame_float, kernel_size=wiener_kernel)
            gm = gpu_proc.gamma_correction(wi, gamma=gamma)
            cl = gpu_proc.apply_clahe_gpu(gm, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
            unsharp = gpu_proc.apply_unsharp_mask_gpu(cl)
            win = gpu_proc.apply_windowing_gpu(unsharp)
            return win
    except Exception as e:
        print(f"GPU processing failed, falling back to CPU: {e}")
        # Complete CPU fallback
        frame_float = frame.astype(np.float32) / 65535.0
        return frame_float

# =========================================================
# Optimized Registration with Grid-based Approach
# =========================================================

def estimate_affine_transform_optimized(src_img, dst_img, method='ecc', use_pyramid=True):
    """Optimized affine transformation with pyramid approach"""
    if method == 'orb':
        src_uint8 = (src_img * 255).astype(np.uint8)
        dst_uint8 = (dst_img * 255).astype(np.uint8)
        
        # Use more keypoints and optimized parameters
        detector = ORB(n_keypoints=3000, fast_threshold=0.03, harris_k=0.04)
        
        detector.detect_and_extract(src_uint8)
        keypoints1 = detector.keypoints
        descriptors1 = detector.descriptors
        
        detector.detect_and_extract(dst_uint8)
        keypoints2 = detector.keypoints
        descriptors2 = detector.descriptors
        
        if descriptors1 is None or descriptors2 is None:
            return np.array([[1, 0, 0], [0, 1, 0]])
        
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.8)
        
        if len(matches) < 4:
            return np.array([[1, 0, 0], [0, 1, 0]])
        
        src_pts = keypoints1[matches[:, 0]]
        dst_pts = keypoints2[matches[:, 1]]
        
        try:
            model_robust, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform,
                min_samples=3,
                residual_threshold=2.0,
                max_trials=2000
            )
            return model_robust.params[:2, :]
        except:
            return np.array([[1, 0, 0], [0, 1, 0]])
    
    elif method == 'ecc':
        src_uint8 = (src_img * 255).astype(np.uint8)
        dst_uint8 = (dst_img * 255).astype(np.uint8)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        if use_pyramid:
            # Multi-scale pyramid approach for better convergence
            scales = [0.25, 0.5, 1.0]
            for scale in scales:
                if scale < 1.0:
                    h, w = src_uint8.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    src_scaled = cv2.resize(src_uint8, (new_w, new_h))
                    dst_scaled = cv2.resize(dst_uint8, (new_w, new_h))
                else:
                    src_scaled = src_uint8
                    dst_scaled = dst_uint8
                
                # Scale the warp matrix
                if scale < 1.0:
                    warp_matrix[:, 2] *= scale
                
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
                
                try:
                    _, warp_matrix = cv2.findTransformECC(
                        dst_scaled, src_scaled, warp_matrix,
                        cv2.MOTION_AFFINE, criteria, None, 5
                    )
                except cv2.error:
                    pass
                
                # Unscale the translation
                if scale < 1.0:
                    warp_matrix[:, 2] /= scale
        else:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    dst_uint8, src_uint8, warp_matrix,
                    cv2.MOTION_AFFINE, criteria
                )
            except cv2.error:
                pass
        
        return warp_matrix
    
    return np.array([[1, 0, 0], [0, 1, 0]])

def apply_affine_transform_gpu(img_float, transform_matrix, use_gpu=True):
    """GPU-accelerated affine transformation"""
    h, w = img_float.shape
    transform_matrix = transform_matrix.astype(np.float32)
    
    if CUDA_AVAILABLE and use_gpu:
        try:
            gpu_img = cv2cuda.GpuMat()
            gpu_img.upload(img_float.astype(np.float32))
            
            gpu_result = cv2cuda.warpAffine(
                gpu_img, transform_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            result = gpu_result.download()
            return np.clip(result, 0, 1)
        except:
            pass
    
    # CPU fallback
    registered = cv2.warpAffine(
        img_float, transform_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return np.clip(registered, 0, 1)

def perform_dsa_gpu(mask_frame, contrast_frame, normalization='standard', use_gpu=True):
    """GPU-accelerated DSA"""
    if use_gpu and GPU_AVAILABLE:
        mask_gpu = cp.asarray(mask_frame)
        contrast_gpu = cp.asarray(contrast_frame)
        
        if normalization == 'standard':
            dsa = contrast_gpu - mask_gpu
        elif normalization == 'weighted':
            epsilon = 1e-6
            dsa = (contrast_gpu - mask_gpu) / (mask_gpu + epsilon)
        elif normalization == 'log':
            epsilon = 1e-6
            dsa = cp.log(mask_gpu + epsilon) - cp.log(contrast_gpu + epsilon)
        else:
            dsa = contrast_gpu - mask_gpu
        
        dsa_min, dsa_max = float(cp.min(dsa)), float(cp.max(dsa))
        if dsa_max > dsa_min:
            dsa_normalized = (dsa - dsa_min) / (dsa_max - dsa_min)
        else:
            dsa_normalized = cp.zeros_like(dsa)
        
        return cp.asnumpy(cp.clip(dsa_normalized, 0, 1))
    else:
        # CPU version
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
# Parallel Processing Thread
# =========================================================

class DSAProcessingThread(QThread):
    """Optimized background thread with parallel processing"""
    progress = Signal(int, str)
    finished = Signal(list, list, list)
    
    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames
        self.params = params
        self.use_gpu = params.get('use_gpu', GPU_AVAILABLE)
        self.num_workers = params.get('num_workers', max(1, mp.cpu_count() - 1))
        
    def run(self):
        try:
            frames_preprocessed = []
            frames_registered = []
            frames_dsa = []
            
            # Step 1: Parallel preprocessing
            self.progress.emit(10, "Preprocessing frames in parallel...")
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                preprocess_func = partial(
                    process_frame_gpu,
                    gamma=self.params['gamma'],
                    clahe_clip=self.params['clahe_clip'],
                    clahe_tiles=self.params['clahe_tiles'],
                    wiener_kernel=self.params['wiener_kernel'],
                    use_gpu=self.use_gpu
                )
                
                futures = [executor.submit(preprocess_func, frame) for frame in self.frames]
                
                for i, future in enumerate(futures):
                    frames_preprocessed.append(future.result())
                    progress_pct = 10 + int(30 * (i + 1) / len(self.frames))
                    self.progress.emit(progress_pct, f"Preprocessed {i+1}/{len(self.frames)}")
            
            # Step 2: Get mask frame
            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_preprocessed[mask_idx]
            self.progress.emit(45, f"Using frame {mask_idx} as mask")
            
            # Step 3: Parallel registration
            self.progress.emit(50, "Registering frames in parallel...")
            
            def register_frame(idx_frame_tuple):
                i, frame = idx_frame_tuple
                if i == mask_idx:
                    return frame
                else:
                    transform = estimate_affine_transform_optimized(
                        frame, mask_frame,
                        method=self.params['registration_method'],
                        use_pyramid=True
                    )
                    return apply_affine_transform_gpu(frame, transform, use_gpu=self.use_gpu)
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                frame_tuples = list(enumerate(frames_preprocessed))
                futures = [executor.submit(register_frame, ft) for ft in frame_tuples]
                
                for i, future in enumerate(futures):
                    frames_registered.append(future.result())
                    progress_pct = 50 + int(30 * (i + 1) / len(self.frames))
                    self.progress.emit(progress_pct, f"Registered {i+1}/{len(self.frames)}")
            
            # Step 4: Parallel DSA
            self.progress.emit(85, "Performing DSA in parallel...")
            
            def compute_dsa(idx_frame_tuple):
                i, contrast_frame = idx_frame_tuple
                if i == mask_idx:
                    return np.zeros_like(mask_frame)
                else:
                    return perform_dsa_gpu(
                        mask_frame, contrast_frame,
                        normalization=self.params['dsa_normalization'],
                        use_gpu=self.use_gpu
                    )
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                frame_tuples = list(enumerate(frames_registered))
                futures = [executor.submit(compute_dsa, ft) for ft in frame_tuples]
                
                for i, future in enumerate(futures):
                    frames_dsa.append(future.result())
                    progress_pct = 85 + int(15 * (i + 1) / len(self.frames))
                    self.progress.emit(progress_pct, f"DSA {i+1}/{len(self.frames)}")
            
            self.progress.emit(100, "Processing complete!")
            self.finished.emit(frames_preprocessed, frames_registered, frames_dsa)
            
        except Exception as e:
            self.progress.emit(0, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

# =========================================================
# DICOM Viewer Widget (Updated)
# =========================================================

class DicomViewer(QWidget):
    """DICOM viewer with GPU-accelerated DSA processing"""
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
        self.use_gpu = GPU_AVAILABLE
        self.num_workers = max(1, mp.cpu_count() - 1)
        
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
        row2.addWidget(QLabel("Registration:"))
        self.reg_combo = QComboBox()
        self.reg_combo.addItems(['ORB (Feature)', 'ECC (Intensity)'])
        self.reg_combo.setCurrentText('ECC (Intensity)')
        row2.addWidget(self.reg_combo)
        
        row2.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row2.addWidget(self.mask_spin)
        
        controls_layout.addLayout(row2)
        
        # Row 3: DSA normalization
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("DSA Method:"))
        self.dsa_combo = QComboBox()
        self.dsa_combo.addItems(['Standard', 'Weighted', 'Logarithmic'])
        row3.addWidget(self.dsa_combo)
        
        # GPU checkbox
        self.gpu_checkbox = QCheckBox("Use GPU")
        self.gpu_checkbox.setChecked(GPU_AVAILABLE)
        self.gpu_checkbox.setEnabled(GPU_AVAILABLE or CUDA_AVAILABLE)
        row3.addWidget(self.gpu_checkbox)
        
        controls_layout.addLayout(row3)
        
        # Row 4: Process button
        row4 = QHBoxLayout()
        self.process_btn = QPushButton("🚀 Process DSA (Optimized)")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        row4.addWidget(self.process_btn)
        
        # Workers spinner
        row4.addWidget(QLabel("CPU Threads:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, mp.cpu_count())
        self.workers_spin.setValue(max(1, mp.cpu_count() - 1))
        row4.addWidget(self.workers_spin)
        
        controls_layout.addLayout(row4)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        # Performance info
        perf_info = f"GPU: {'✓ Available' if GPU_AVAILABLE else '✗ Not Available'} | "
        perf_info += f"CUDA: {'✓ Available' if CUDA_AVAILABLE else '✗ Not Available'} | "
        perf_info += f"CPU Cores: {mp.cpu_count()}"
        self.perf_label = QLabel(perf_info)
        self.perf_label.setStyleSheet("color: blue; font-size: 9px;")
        controls_layout.addWidget(self.perf_label)
        
        layout.addWidget(controls_widget)

    def show_save_menu(self):
        """Show save options menu"""
        menu = QMenu(self)
        
        menu.addAction("Save Current Frame as PNG", lambda: self.save_current_frame('png'))
        menu.addAction("Save Current Frame as TIFF (16-bit)", lambda: self.save_current_frame('tiff'))
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
        """Start optimized DSA processing in background thread"""
        if self.frames is None:
            return
        
        # Get parameters
        params = {
            'gamma': self.gamma_spin.value(),
            'clahe_clip': self.clahe_spin.value(),
            'clahe_tiles': (8, 8),
            'wiener_kernel': self.wiener_spin.value(),
            'mask_frame_index': self.mask_spin.value(),
            'registration_method': 'orb' if 'ORB' in self.reg_combo.currentText() else 'ecc',
            'dsa_normalization': self.dsa_combo.currentText().lower(),
            'use_gpu': self.gpu_checkbox.isChecked(),
            'num_workers': self.workers_spin.value()
        }
        
        # Update stored parameters
        self.gamma = params['gamma']
        self.clahe_clip = params['clahe_clip']
        self.wiener_kernel = params['wiener_kernel']
        self.mask_frame_index = params['mask_frame_index']
        self.registration_method = params['registration_method']
        self.dsa_normalization = params['dsa_normalization']
        self.use_gpu = params['use_gpu']
        self.num_workers = params['num_workers']
        
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
        self.status_label.setText("Status: Processing Complete! ✓")
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

class Desktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU-Accelerated DSA Viewer - High Performance")
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
        
        # Show GPU/CPU info
        if GPU_AVAILABLE:
            gpu_info = "🚀 GPU Acceleration: ENABLED"
        elif CUDA_AVAILABLE:
            gpu_info = "⚡ CUDA Available"
        else:
            gpu_info = "💻 CPU Mode"
        
        self.gpu_status = QLabel(gpu_info)
        
        self.statusBar().addWidget(self.status_time)
        self.statusBar().addWidget(self.gpu_status)
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
            
            # Update viewers
            short_name = os.path.basename(filename)
            self.viewer1.set_frames(frames, short_name)
            self.viewer2.set_frames(frames, short_name)
            
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
    
    # Print system info
    print("\n" + "="*60)
    print("GPU-ACCELERATED DSA VIEWER - PERFORMANCE MODE")
    print("="*60)
    print(f"GPU (CuPy):        {'✓ ENABLED' if GPU_AVAILABLE else '✗ Not Available'}")
    print(f"CUDA (OpenCV):     {'✓ ENABLED' if CUDA_AVAILABLE else '✗ Not Available'}")
    print(f"CPU Cores:         {mp.cpu_count()}")
    print(f"Default Workers:   {max(1, mp.cpu_count() - 1)}")
    print("="*60 + "\n")
    
    window = Desktop()
    window.show()
    
    app.exec()