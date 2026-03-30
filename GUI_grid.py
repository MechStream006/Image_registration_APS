import os
import numpy as np
import pydicom
import cv2
from PySide6.QtCore import Qt, QTimer, QTime, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter, QToolBar,
    QComboBox, QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMenu, QGroupBox
)
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RBFInterpolator

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
# Grid Registration Functions
# =========================================================

def create_feather_mask(h, w, feather_size=20):
    """Create a mask with smooth edges (feathering)"""
    mask = np.ones((h, w), dtype=np.float32)
    feather_size = min(feather_size, h//4, w//4)
    
    if feather_size > 0:
        for i in range(feather_size):
            weight = i / feather_size
            if i < h:
                mask[i, :] *= weight
                mask[h - 1 - i, :] *= weight
        
        for j in range(feather_size):
            weight = j / feather_size
            if j < w:
                mask[:, j] *= weight
                mask[:, w - 1 - j] *= weight
    
    return mask

def is_valid_transform(transform, max_scale=2.0, max_translation=50):
    """Check if transformation is reasonable"""
    if not np.isfinite(transform).all():
        return False
    
    scale_x = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)
    scale_y = np.sqrt(transform[0, 1]**2 + transform[1, 1]**2)
    tx = abs(transform[0, 2])
    ty = abs(transform[1, 2])
    
    if scale_x < 1/max_scale or scale_x > max_scale:
        return False
    if scale_y < 1/max_scale or scale_y > max_scale:
        return False
    if tx > max_translation or ty > max_translation:
        return False
    
    return True

def estimate_single_ecc(src_patch, dst_patch):
    """Estimate ECC transformation for a single patch"""
    src_uint8 = (src_patch * 255).astype(np.uint8)
    dst_uint8 = (dst_patch * 255).astype(np.uint8)
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    
    try:
        _, warp_matrix = cv2.findTransformECC(
            dst_uint8, src_uint8, warp_matrix,
            cv2.MOTION_AFFINE, criteria
        )
    except cv2.error:
        pass
    
    return warp_matrix

def estimate_single_orb(src_patch, dst_patch):
    """Estimate ORB transformation for a single patch"""
    src_uint8 = (src_patch * 255).astype(np.uint8)
    dst_uint8 = (dst_patch * 255).astype(np.uint8)
    
    detector = ORB(n_keypoints=2000, fast_threshold=0.05)
    
    detector.detect_and_extract(src_uint8)
    keypoints1 = detector.keypoints
    descriptors1 = detector.descriptors
    
    detector.detect_and_extract(dst_uint8)
    keypoints2 = detector.keypoints
    descriptors2 = detector.descriptors
    
    if len(keypoints1) < 4 or len(keypoints2) < 4:
        return np.array([[1, 0, 0], [0, 1, 0]])
    
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    
    if len(matches) < 4:
        return np.array([[1, 0, 0], [0, 1, 0]])
    
    src_pts = keypoints1[matches[:, 0]]
    dst_pts = keypoints2[matches[:, 1]]
    
    try:
        model_robust, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform,
            min_samples=3,
            residual_threshold=2,
            max_trials=1000
        )
        return model_robust.params[:2, :]
    except:
        return np.array([[1, 0, 0], [0, 1, 0]])

class GridRegistrationEngine:
    """Advanced grid-based registration engine"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
    
    def estimate_grid_transforms(self, src_img, dst_img, grid_size=(4, 4), 
                                method='ecc', overlap=0.25, 
                                use_parallel=True, progress_callback=None):
        """
        Estimate transformations for grid cells
        
        Args:
            src_img: Source image (float, 0-1)
            dst_img: Destination image (float, 0-1)
            grid_size: (rows, cols) grid divisions
            method: 'ecc' or 'orb'
            overlap: Overlap fraction for extended patches
            use_parallel: Use parallel processing
            progress_callback: Function(current, total, message)
        
        Returns:
            List of transform dictionaries
        """
        h, w = src_img.shape
        grid_h, grid_w = grid_size
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        overlap_h = int(cell_h * overlap)
        overlap_w = int(cell_w * overlap)
        
        # Create grid tasks
        grid_tasks = []
        for i in range(grid_h):
            for j in range(grid_w):
                # Extended boundaries with overlap
                y1 = max(0, i * cell_h - overlap_h)
                y2 = min(h, (i + 1) * cell_h + overlap_h)
                x1 = max(0, j * cell_w - overlap_w)
                x2 = min(w, (j + 1) * cell_w + overlap_w)
                
                # Core region
                core_y1 = i * cell_h
                core_y2 = (i + 1) * cell_h if i < grid_h - 1 else h
                core_x1 = j * cell_w
                core_x2 = (j + 1) * cell_w if j < grid_w - 1 else w
                
                grid_tasks.append({
                    'idx': i * grid_w + j,
                    'bbox': (x1, y1, x2, y2),
                    'core_bbox': (core_x1, core_y1, core_x2, core_y2),
                    'center': ((core_x1 + core_x2) // 2, (core_y1 + core_y2) // 2)
                })
        
        transforms = [None] * len(grid_tasks)
        
        if use_parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {}
                for task in grid_tasks:
                    future = executor.submit(
                        self._process_single_grid,
                        src_img, dst_img, task, method
                    )
                    future_to_idx[future] = task['idx']
                
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        transforms[idx] = future.result()
                    except Exception as e:
                        task = grid_tasks[idx]
                        transforms[idx] = self._create_identity_transform(task)
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(grid_tasks), 
                                        f"Processing grid {completed}/{len(grid_tasks)}")
        else:
            # Sequential processing
            for i, task in enumerate(grid_tasks):
                try:
                    transforms[task['idx']] = self._process_single_grid(
                        src_img, dst_img, task, method
                    )
                except Exception as e:
                    transforms[task['idx']] = self._create_identity_transform(task)
                
                if progress_callback:
                    progress_callback(i + 1, len(grid_tasks), 
                                    f"Processing grid {i+1}/{len(grid_tasks)}")
        
        # Interpolate missing/invalid transforms
        transforms = self._interpolate_missing_transforms(transforms, grid_size)
        
        return transforms
    
    def _process_single_grid(self, src_img, dst_img, task, method):
        """Process a single grid cell"""
        x1, y1, x2, y2 = task['bbox']
        
        src_patch = src_img[y1:y2, x1:x2]
        dst_patch = dst_img[y1:y2, x1:x2]
        
        # Check for sufficient content
        src_var = src_patch.std()
        dst_var = dst_patch.std()
        
        if src_var < 0.01 or dst_var < 0.01:
            transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            quality = 'low_content'
        else:
            # Estimate transformation
            if method == 'ecc':
                transform = estimate_single_ecc(src_patch, dst_patch)
            else:  # orb
                transform = estimate_single_orb(src_patch, dst_patch)
            
            # Validate transform
            if is_valid_transform(transform):
                quality = 'good'
            else:
                transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                quality = 'invalid'
        
        return {
            'transform': transform,
            'bbox': task['bbox'],
            'core_bbox': task['core_bbox'],
            'center': task['center'],
            'quality': quality,
            'variance': src_var
        }
    
    def _create_identity_transform(self, task):
        """Create identity transform for failed grids"""
        return {
            'transform': np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            'bbox': task['bbox'],
            'core_bbox': task['core_bbox'],
            'center': task['center'],
            'quality': 'failed',
            'variance': 0.0
        }
    
    def _interpolate_missing_transforms(self, transforms, grid_size):
        """Interpolate transforms for low-quality regions"""
        grid_h, grid_w = grid_size
        
        for i in range(grid_h):
            for j in range(grid_w):
                idx = i * grid_w + j
                
                if transforms[idx]['quality'] in ['low_content', 'invalid', 'failed']:
                    # Collect valid neighbors
                    valid_neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_h and 0 <= nj < grid_w:
                                neighbor_idx = ni * grid_w + nj
                                if transforms[neighbor_idx]['quality'] == 'good':
                                    valid_neighbors.append(transforms[neighbor_idx]['transform'])
                    
                    if valid_neighbors:
                        avg_transform = np.mean(valid_neighbors, axis=0)
                        transforms[idx]['transform'] = avg_transform
                        transforms[idx]['quality'] = 'interpolated'
        
        return transforms
    
    def apply_grid_transforms(self, img_float, transforms, method='gaussian'):
        """
        Apply grid transformations to image
        
        Args:
            img_float: Input image
            transforms: List of transform dictionaries
            method: 'gaussian' or 'rbf'
        """
        if method == 'rbf':
            return self._apply_rbf_transforms(img_float, transforms)
        else:
            return self._apply_gaussian_transforms(img_float, transforms)
    
    def _apply_gaussian_transforms(self, img_float, transforms):
        """Apply with Gaussian-weighted blending"""
        h, w = img_float.shape
        result = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        yy, xx = np.ogrid[0:h, 0:w]
        
        for trans_dict in transforms:
            transform = trans_dict['transform']
            cx, cy = trans_dict['center']
            cx1, cy1, cx2, cy2 = trans_dict['core_bbox']
            
            # Calculate standard deviation
            sigma_x = max((cx2 - cx1) / 2, 1)
            sigma_y = max((cy2 - cy1) / 2, 1)
            
            # Create Gaussian weight map
            weight = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + 
                             (yy - cy)**2 / (2 * sigma_y**2)))
            
            # Only apply within reasonable distance
            mask = (np.abs(xx - cx) < 3 * sigma_x) & (np.abs(yy - cy) < 3 * sigma_y)
            weight = weight * mask
            
            # Transform entire image
            transformed = cv2.warpAffine(
                img_float, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Accumulate
            result += transformed * weight
            weight_sum += weight
        
        # Normalize
        result = np.divide(result, weight_sum, where=weight_sum > 1e-6, out=result)
        
        return np.clip(result, 0, 1)
    
    def _apply_rbf_transforms(self, img_float, transforms):
        """Apply with RBF interpolation for smooth deformation field"""
        h, w = img_float.shape
        
        # Extract control points and displacements
        control_points = []
        displacements_x = []
        displacements_y = []
        
        for trans_dict in transforms:
            if trans_dict['quality'] != 'failed':
                cx, cy = trans_dict['center']
                transform = trans_dict['transform']
                
                # Apply transform to center point
                point = np.array([cx, cy, 1])
                transformed = transform @ point
                
                dx = transformed[0] - cx
                dy = transformed[1] - cy
                
                control_points.append([cx, cy])
                displacements_x.append(dx)
                displacements_y.append(dy)
        
        if len(control_points) < 3:
            # Not enough points for RBF, return original
            return img_float
        
        control_points = np.array(control_points)
        displacements_x = np.array(displacements_x)
        displacements_y = np.array(displacements_y)
        
        # Add boundary control points (identity)
        boundary_points = [
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [w//2, h-1], [0, h//2], [w-1, h//2]
        ]
        control_points = np.vstack([control_points, boundary_points])
        displacements_x = np.concatenate([displacements_x, [0]*len(boundary_points)])
        displacements_y = np.concatenate([displacements_y, [0]*len(boundary_points)])
        
        # Create RBF interpolators
        try:
            rbf_x = RBFInterpolator(control_points, displacements_x, 
                                   kernel='thin_plate_spline', smoothing=0.1)
            rbf_y = RBFInterpolator(control_points, displacements_y, 
                                   kernel='thin_plate_spline', smoothing=0.1)
            
            # Generate dense deformation field
            yy, xx = np.mgrid[0:h, 0:w]
            query_points = np.column_stack([xx.ravel(), yy.ravel()])
            
            dx_dense = rbf_x(query_points).reshape(h, w)
            dy_dense = rbf_y(query_points).reshape(h, w)
            
            # Create mapping arrays
            map_x = (xx + dx_dense).astype(np.float32)
            map_y = (yy + dy_dense).astype(np.float32)
            
            # Apply deformation
            result = cv2.remap(img_float, map_x, map_y, 
                             cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
            
            return np.clip(result, 0, 1)
        except:
            # RBF failed, fallback to Gaussian
            return self._apply_gaussian_transforms(img_float, transforms)

# =========================================================
# Standard Registration Functions
# =========================================================

def estimate_affine_transform(src_img, dst_img, method='orb'):
    """Estimate affine transformation between two images (global)"""
    if method == 'orb':
        src_uint8 = (src_img * 255).astype(np.uint8)
        dst_uint8 = (dst_img * 255).astype(np.uint8)
        
        detector = ORB(n_keypoints=2000, fast_threshold=0.05)
        
        detector.detect_and_extract(src_uint8)
        keypoints1 = detector.keypoints
        descriptors1 = detector.descriptors
        
        detector.detect_and_extract(dst_uint8)
        keypoints2 = detector.keypoints
        descriptors2 = detector.descriptors
        
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
        
        if len(matches) < 4:
            return np.array([[1, 0, 0], [0, 1, 0]])
        
        src_pts = keypoints1[matches[:, 0]]
        dst_pts = keypoints2[matches[:, 1]]
        
        model_robust, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform,
            min_samples=3,
            residual_threshold=2,
            max_trials=1000
        )
        
        return model_robust.params[:2, :]
    
    elif method == 'ecc':
        src_uint8 = (src_img * 255).astype(np.uint8)
        dst_uint8 = (dst_img * 255).astype(np.uint8)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
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

def apply_affine_transform(img_float, transform_matrix):
    """Apply affine transformation to image (global)"""
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
# Processing Thread
# =========================================================

class DSAProcessingThread(QThread):
    """Background thread for DSA processing"""
    progress = Signal(int, str)
    finished = Signal(list, list, list)
    
    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames
        self.params = params
        self.grid_engine = GridRegistrationEngine(max_workers=params.get('parallel_workers', 4))
        
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
                progress_pct = 10 + int(25 * (i + 1) / len(self.frames))
                self.progress.emit(progress_pct, f"Preprocessing frame {i+1}/{len(self.frames)}")
            
            # Step 2: Get mask frame
            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_preprocessed[mask_idx]
            self.progress.emit(40, f"Using frame {mask_idx} as mask")
            
            # Step 3: Register frames
            self.progress.emit(45, "Registering frames...")
            use_grid = self.params.get('use_grid_registration', False)
            
            for i, frame in enumerate(frames_preprocessed):
                if i == mask_idx:
                    frames_registered.append(frame)
                else:
                    if use_grid:
                        # Grid-based registration
                        def grid_progress(curr, total, msg):
                            base = 45 + int(35 * i / len(self.frames))
                            pct = base + int(35 * curr / (total * len(self.frames)))
                            self.progress.emit(pct, f"Frame {i+1}: {msg}")
                        
                        transforms = self.grid_engine.estimate_grid_transforms(
                            frame, mask_frame,
                            grid_size=self.params['grid_size'],
                            method=self.params['registration_method'],
                            overlap=0.25,
                            use_parallel=self.params.get('use_parallel', True),
                            progress_callback=grid_progress
                        )
                        
                        registered = self.grid_engine.apply_grid_transforms(
                            frame, transforms,
                            method=self.params.get('grid_blend_method', 'gaussian')
                        )
                    else:
                        # Global registration
                        transform = estimate_affine_transform(
                            frame, mask_frame, 
                            method=self.params['registration_method']
                        )
                        registered = apply_affine_transform(frame, transform)
                    
                    frames_registered.append(registered)
                
                if not use_grid:
                    progress_pct = 45 + int(35 * (i + 1) / len(self.frames))
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
            self.progress.emit(0, f"Error: {str(e)}")

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
        self.use_grid_registration = False
        self.grid_size = (4, 4)
        self.grid_blend_method = 'gaussian'
        self.use_parallel = True
        self.parallel_workers = 4
        
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
        title = "DSA Viewer (Advanced Grid Registration)" if dsa_mode else "Original DICOM"
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
        self.reg_combo.addItems(['ORB (Feature-based)', 'ECC (Intensity-based)'])
        self.reg_combo.setCurrentText('ECC (Intensity-based)')
        row2.addWidget(self.reg_combo)
        
        row2.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row2.addWidget(self.mask_spin)
        
        controls_layout.addLayout(row2)
        
        # Row 3: Grid registration options
        grid_group = QGroupBox("Grid Registration (Advanced)")
        grid_layout = QVBoxLayout(grid_group)
        
        grid_row1 = QHBoxLayout()
        self.use_grid_check = QCheckBox("Enable Grid Registration")
        self.use_grid_check.setChecked(False)
        grid_row1.addWidget(self.use_grid_check)
        
        grid_row1.addWidget(QLabel("Grid Size:"))
        self.grid_rows_spin = QSpinBox()
        self.grid_rows_spin.setRange(2, 8)
        self.grid_rows_spin.setValue(4)
        self.grid_rows_spin.setPrefix("Rows: ")
        grid_row1.addWidget(self.grid_rows_spin)
        
        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(2, 8)
        self.grid_cols_spin.setValue(4)
        self.grid_cols_spin.setPrefix("Cols: ")
        grid_row1.addWidget(self.grid_cols_spin)
        grid_layout.addLayout(grid_row1)
        
        grid_row2 = QHBoxLayout()
        grid_row2.addWidget(QLabel("Blend Method:"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(['Gaussian', 'RBF (Smooth)'])
        grid_row2.addWidget(self.blend_combo)
        
        self.parallel_check = QCheckBox("Parallel Processing")
        self.parallel_check.setChecked(True)
        grid_row2.addWidget(self.parallel_check)
        grid_layout.addLayout(grid_row2)
        
        controls_layout.addWidget(grid_group)
        
        # Row 4: DSA normalization
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("DSA Method:"))
        self.dsa_combo = QComboBox()
        self.dsa_combo.addItems(['Standard', 'Weighted', 'Logarithmic'])
        row4.addWidget(self.dsa_combo)
        
        self.process_btn = QPushButton("🔄 Process DSA Pipeline")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        row4.addWidget(self.process_btn)
        
        controls_layout.addLayout(row4)
        
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
            'registration_method': 'orb' if 'ORB' in self.reg_combo.currentText() else 'ecc',
            'dsa_normalization': self.dsa_combo.currentText().lower(),
            'use_grid_registration': self.use_grid_check.isChecked(),
            'grid_size': (self.grid_rows_spin.value(), self.grid_cols_spin.value()),
            'grid_blend_method': 'rbf' if 'RBF' in self.blend_combo.currentText() else 'gaussian',
            'use_parallel': self.parallel_check.isChecked(),
            'parallel_workers': 4
        }
        
        # Update stored parameters
        self.gamma = params['gamma']
        self.clahe_clip = params['clahe_clip']
        self.wiener_kernel = params['wiener_kernel']
        self.mask_frame_index = params['mask_frame_index']
        self.registration_method = params['registration_method']
        self.dsa_normalization = params['dsa_normalization']
        self.use_grid_registration = params['use_grid_registration']
        self.grid_size = params['grid_size']
        self.grid_blend_method = params['grid_blend_method']
        
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

class Desktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced DSA Viewer - Grid Registration Edition")
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
        
        self.open_btn = QPushButton("📂 Open DICOM")
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
    
    window = Desktop()
    window.show()
    
    app.exec()