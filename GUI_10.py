import os
import numpy as np
import pydicom
import cv2
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter, QToolBar,
    QComboBox, QSlider, QCheckBox, QSpinBox, QProgressBar, QMenu, QGroupBox
)
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid

# =========================================================
# GPU detection (used by histogram matching)
# =========================================================
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU (CuPy) available - GPU acceleration ENABLED")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - Running in CPU mode")

def clear_gpu_cache():
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

def get_gpu_memory_info():
    if not GPU_AVAILABLE:
        return
    mempool = cp.get_default_memory_pool()
    print(f"GPU Memory Used: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"GPU Memory Total: {mempool.total_bytes() / 1e9:.2f} GB")

# =========================================================
# Helpers: float conversion
# =========================================================
def to_float_unit(img):
    """Convert image to float32 in [0,1] depending on dtype."""
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    # Fallback: min-max normalize
    a = img.astype(np.float32)
    amin, amax = np.min(a), np.max(a)
    if amax == amin:
        return np.zeros_like(a, dtype=np.float32)
    return (a - amin) / (amax - amin)

def from_float_unit(img_float):
    """Convert float32 [0,1] image to uint16."""
    return np.clip(img_float * 65535.0, 0, 65535).astype(np.uint16)

# =========================================================
# Histogram Matching (optional, GPU-accelerated if available)
# =========================================================
def histogram_matching_gpu(source_uint16, reference_uint16, bins=256):
    """
    Match histogram of 'source' to 'reference' (16-bit). Returns float32 in [0,1].
    """
    DICOM_PIXEL_MAX = 65535.0

    if GPU_AVAILABLE:
        source_gpu = cp.asarray(source_uint16.astype(np.float32))
        reference_gpu = cp.asarray(reference_uint16.astype(np.float32))

        src_hist, src_bins = cp.histogram(source_gpu.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))
        ref_hist, ref_bins = cp.histogram(reference_gpu.ravel(), bins=bins, range=(0, DICOM_PIXEL_MAX))

        src_hist = src_hist.astype(cp.float32) + 1e-6
        ref_hist = ref_hist.astype(cp.float32) + 1e-6

        src_cdf = cp.cumsum(src_hist); src_cdf /= src_cdf[-1]
        ref_cdf = cp.cumsum(ref_hist); ref_cdf /= ref_cdf[-1]

        src_centers = src_bins[:-1] + (src_bins[1] - src_bins[0]) / 2.0
        ref_centers = ref_bins[:-1] + (ref_bins[1] - ref_bins[0]) / 2.0

        lut = cp.interp(src_cdf, ref_cdf, ref_centers)

        idx = cp.searchsorted(src_centers, source_gpu.ravel(), side='left')
        idx = cp.clip(idx, 0, len(lut) - 1)

        matched = lut[idx].reshape(source_gpu.shape)
        result = cp.clip(matched, 0, DICOM_PIXEL_MAX).astype(cp.float32)
        return (cp.asnumpy(result) / DICOM_PIXEL_MAX).astype(np.float32)
    else:
        src = source_uint16.astype(np.uint16)
        ref = reference_uint16.astype(np.uint16)

        src_hist, src_bins = np.histogram(src.ravel(), bins=bins, range=(0, int(DICOM_PIXEL_MAX)))
        ref_hist, ref_bins = np.histogram(ref.ravel(), bins=bins, range=(0, int(DICOM_PIXEL_MAX)))

        src_cdf = np.cumsum(src_hist).astype(np.float32); src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float32); ref_cdf /= ref_cdf[-1]

        lut = np.interp(src_cdf, ref_cdf, ref_bins[:-1])  # map CDF->intensity
        matched = np.interp(src.ravel(), src_bins[:-1], lut).reshape(src.shape).astype(np.float32)
        return np.clip(matched / DICOM_PIXEL_MAX, 0, 1).astype(np.float32)

# =========================================================
# Phase Correlation (PCC) Registration
# =========================================================
def _prep_for_fft(img, use_gpu=False):
    img = img.astype(np.float32)
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
    Compute phase correlation shift between two images (expects uint16 or float convertible).
    Returns (dy, dx).
    """
    a = _prep_for_fft(mask_frame.astype(np.float32))
    b = _prep_for_fft(live_frame.astype(np.float32))

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
    h, w = img_float.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    warped = cv2.warpAffine(img_float, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0, 1)

# =========================================================
# ECC Registration (global affine/rigid/translation)
# =========================================================
def ecc_registration(mask_frame, live_frame,
                     motion_model=cv2.MOTION_AFFINE,
                     num_levels=3,
                     max_iterations=1000,
                     epsilon=1e-6,
                     verbose=False):
    """
    ECC (Enhanced Correlation Coefficient) image registration.
    Inputs: float32 arrays (any dynamic range) — internally normalized.
    Returns: (warp_matrix 2x3, cc_value)
    """
    ref = mask_frame.astype(np.float32)
    mov = live_frame.astype(np.float32)

    # Normalize each to 0..1, then z-score (helps ECC)
    ref = cv2.normalize(ref, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mov = cv2.normalize(mov, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    ref -= np.mean(ref); mov -= np.mean(mov)
    ref /= (np.std(ref) + 1e-8); mov /= (np.std(mov) + 1e-8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)

    try:
        cc_value, warp_matrix = cv2.findTransformECC(
            ref, mov, warp_matrix, motion_model, criteria, None, num_levels
        )
        if verbose:
            print(f"✅ ECC converged: corr={cc_value:.6f}")
    except cv2.error as e:
        print(f"⚠️ ECC registration failed: {e}")
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        cc_value = 0.0

    return warp_matrix, cc_value

def apply_affine_warp(img_float, transform_matrix):
    h, w = img_float.shape
    transform_matrix = transform_matrix.astype(np.float32)
    registered = cv2.warpAffine(img_float, transform_matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    return np.clip(registered, 0, 1)

# =========================================================
# Local (grid) ECC Registration (translation per tile)
# =========================================================
def divide_into_tiles(img, grid_size=(8, 8)):
    h, w = img.shape
    gh, gw = grid_size
    tile_h, tile_w = h // gh, w // gw
    tiles = []
    for i in range(gh):
        for j in range(gw):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            tiles.append((y0, y1, x0, x1))
    return tiles

def local_ecc_registration(mask_frame, live_frame, grid_size=(8, 8)):
    """
    Piecewise translation ECC. Returns dense flow field (H,W,2) as (dx, dy).
    """
    h, w = mask_frame.shape
    flow_field = np.zeros((h, w, 2), dtype=np.float32)
    tiles = divide_into_tiles(mask_frame, grid_size)

    for (y0, y1, x0, x1) in tiles:
        ref_tile = mask_frame[y0:y1, x0:x1]
        mov_tile = live_frame[y0:y1, x0:x1]

        if ref_tile.size < 900:  # very small tiles or degenerate
            continue
        if np.std(ref_tile) < 1e-5 or np.std(mov_tile) < 1e-5:
            continue

        try:
            warp, cc = ecc_registration(
                ref_tile, mov_tile,
                motion_model=cv2.MOTION_TRANSLATION,
                num_levels=0,
                max_iterations=200,
                epsilon=1e-5
            )
            dx, dy = warp[0, 2], warp[1, 2]
            flow_field[y0:y1, x0:x1, 0] = dx
            flow_field[y0:y1, x0:x1, 1] = dy
        except cv2.error as e:
            print(f"⚠️ ECC skipped tile ({y0},{x0}): {str(e)[:80]}")
            continue

    # Smooth across tile boundaries
    flow_field[..., 0] = cv2.GaussianBlur(flow_field[..., 0], (9, 9), 1.5)
    flow_field[..., 1] = cv2.GaussianBlur(flow_field[..., 1], (9, 9), 1.5)
    return flow_field

def warp_with_flow(img, flow):
    """
    Apply nonrigid warp from dense flow (dx, dy per pixel).
    """
    h, w = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(img.astype(np.float32), map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0, 1)

# =========================================================
# DSA subtraction
# =========================================================
def perform_dsa(mask_frame_float, contrast_frame_float):
    dsa = contrast_frame_float - mask_frame_float
    dsa_min, dsa_max = float(dsa.min()), float(dsa.max())
    if dsa_max > dsa_min:
        dsa_norm = (dsa - dsa_min) / (dsa_max - dsa_min)
    else:
        dsa_norm = np.zeros_like(dsa, dtype=np.float32)
    return np.clip(dsa_norm, 0, 1)

# =========================================================
# Processing Thread (no pre/post processing)
# =========================================================
class DSAProcessingThread(QThread):
    progress = Signal(int, str)
    finished = Signal(list, list)  # (registered_frames_float, dsa_frames_float)

    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames              # original frames as loaded (numpy arrays)
        self.params = params

    def run(self):
        try:
            self.progress.emit(5, "Preparing frames...")
            # Convert all frames to float [0,1] once
            frames_float = [to_float_unit(f) for f in self.frames]
            mask_idx = self.params['mask_frame_index']
            mask_frame = frames_float[mask_idx]
            mask_u16 = from_float_unit(mask_frame)

            frames_registered = []
            frames_dsa = []

            self.progress.emit(15, "Registering frames...")

            method = self.params['registration_method']
            total = len(frames_float)
            for i, frame in enumerate(frames_float):
                if i == mask_idx:
                    registered = frame
                else:
                    if method == 'pcc':
                        # PCC expects comparable intensity; use uint16 for stability
                        dy, dx = phase_correlation_registration(mask_u16, from_float_unit(frame))
                        registered = apply_translation_warp(frame, dy, dx)
                    elif method == 'ecc_local':
                        flow = local_ecc_registration(mask_frame, frame, grid_size=(8, 8))
                        registered = warp_with_flow(frame, flow)
                    else:
                        motion_map = {
                            'ecc_affine': cv2.MOTION_AFFINE,
                            'ecc_euclidean': cv2.MOTION_EUCLIDEAN,
                            'ecc_translation': cv2.MOTION_TRANSLATION
                        }
                        model = motion_map.get(method, cv2.MOTION_AFFINE)
                        transform, cc_val = ecc_registration(mask_frame, frame, motion_model=model)
                        if cc_val < 0.5:
                            print(f"⚠️ Low ECC correlation ({cc_val:.3f}) — registration may be inaccurate.")
                        registered = apply_affine_warp(frame, transform)

                    # Optional histogram matching (registered -> mask)
                    if self.params['enable_histogram_matching']:
                        registered = histogram_matching_gpu(from_float_unit(registered), mask_u16)

                frames_registered.append(registered)
                self.progress.emit(15 + int(55 * (i + 1) / total),
                                   f"Registering frame {i+1}/{total}")

            # DSA
            self.progress.emit(75, "Performing DSA subtraction...")
            for i, contrast in enumerate(frames_registered):
                if i == mask_idx:
                    dsa_frame = np.zeros_like(mask_frame, dtype=np.float32)
                else:
                    dsa_frame = perform_dsa(mask_frame, contrast)
                frames_dsa.append(dsa_frame)
                self.progress.emit(75 + int(25 * (i + 1) / total),
                                   f"Computing DSA {i+1}/{total}")

            self.progress.emit(100, "Processing complete!")
            self.finished.emit(frames_registered, frames_dsa)
            clear_gpu_cache()

        except Exception as e:
            self.progress.emit(0, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

# =========================================================
# DICOM Save Utilities
# =========================================================
def save_multiframe_dicom(original_dcm, frames_float, out_path, description_suffix=""):
    frames_scaled = [from_float_unit(f) for f in frames_float]

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

    # New UIDs
    new_dcm.SeriesInstanceUID = generate_uid()
    new_dcm.SOPInstanceUID = generate_uid()

    new_dcm.PixelData = np.stack(frames_scaled).tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    print(f"✅ Saved multi-frame DICOM (16-bit): {out_path}")

# =========================================================
# DICOM Viewer Widget (kept; no pre/post controls)
# =========================================================
class DicomViewer(QWidget):
    """DICOM viewer with optional DSA visualization."""
    def __init__(self, parent=None, dsa_mode=False):
        super().__init__(parent)
        self.frames = None
        self.frame_index = 0
        self.dsa_mode = dsa_mode
        self.original_dicom = None

        # Processing parameters (only what remains)
        self.mask_frame_index = 0
        self.registration_method = 'pcc'
        self.enable_histogram_matching = True

        # Processed cache
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
        title = "DSA Viewer" if dsa_mode else "Original DICOM"
        self.image_label = QLabel(title)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label, 1)

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
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # ========== REGISTRATION GROUP ==========
        registration_group = QGroupBox("Registration Parameters")
        registration_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Registration Method:"))
        self.reg_combo = QComboBox()
        self.reg_combo.addItems([
            'PCC (Phase Correlation)',
            'ECC (Affine)',
            'ECC (Euclidean)',
            'ECC (Translation)',
            'ECC (Local Grid)'
        ])
        self.reg_combo.setCurrentText('PCC (Phase Correlation)')
        row1.addWidget(self.reg_combo)

        row1.addWidget(QLabel("Mask Frame:"))
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.setValue(0)
        row1.addWidget(self.mask_spin)

        registration_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.hist_match_checkbox = QCheckBox("Enable Histogram Matching")
        self.hist_match_checkbox.setChecked(True)
        row2.addWidget(self.hist_match_checkbox)
        registration_layout.addLayout(row2)

        registration_group.setLayout(registration_layout)
        controls_layout.addWidget(registration_group)

        # ========== PROCESS BUTTON ==========
        self.process_btn = QPushButton("🔄 Run Registration + DSA")
        self.process_btn.clicked.connect(self.process_dsa_pipeline)
        self.process_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        controls_layout.addWidget(self.process_btn)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        controls_layout.addWidget(self.status_label)

        # GPU status
        if GPU_AVAILABLE:
            gpu_label = QLabel("✅ Histogram Matching GPU: ENABLED")
            gpu_label.setStyleSheet("color: green;")
        else:
            gpu_label = QLabel("⚠️ Histogram Matching GPU: DISABLED (CPU mode)")
            gpu_label.setStyleSheet("color: orange;")
        controls_layout.addWidget(gpu_label)

        layout.addWidget(controls_widget)

    def show_save_menu(self):
        menu = QMenu(self)
        menu.addAction("💾 Save Current DSA Frame (Single DICOM)", self.save_current_frame_dicom)
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
            self, "Save Current DSA Frame",
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
        self.frames = frames
        self.frame_index = 0
        self.original_dicom = original_dicom

        if self.dsa_mode and hasattr(self, 'mask_spin'):
            self.mask_spin.setRange(0, len(frames) - 1)

        if filename:
            self.filename_label.setText(filename)
        self.info_label.setText(f"{len(frames)} frames")

        # Clear processed caches
        self.frames_registered = None
        self.frames_dsa = None

        if self.dsa_mode and hasattr(self, 'save_btn'):
            self.save_btn.setEnabled(False)

        self.show_frame()

    def process_dsa_pipeline(self):
        if self.frames is None:
            return

        # Determine registration method
        reg_text = self.reg_combo.currentText()
        if 'PCC' in reg_text:
            method = 'pcc'
        elif 'Affine' in reg_text:
            method = 'ecc_affine'
        elif 'Euclidean' in reg_text:
            method = 'ecc_euclidean'
        elif 'Translation' in reg_text:
            method = 'ecc_translation'
        else:
            method = 'ecc_local'

        params = {
            'mask_frame_index': self.mask_spin.value(),
            'registration_method': method,
            'enable_histogram_matching': self.hist_match_checkbox.isChecked(),
        }

        # Update stored params
        self.mask_frame_index = params['mask_frame_index']
        self.registration_method = params['registration_method']
        self.enable_histogram_matching = params['enable_histogram_matching']

        # UI state
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Status: Processing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        # Start worker
        self.processing_thread = DSAProcessingThread(self.frames, params)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Status: {message}")

    def on_processing_finished(self, registered, dsa):
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
                img = (self.frames_dsa[self.frame_index] * 255).astype(np.uint8)
            else:
                # Show original frame
                frame = self.frames[self.frame_index]
                img = self._rescale_to_uint8(frame)

            h, w = img.shape
            qimg = QImage(img.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
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
        self.setWindowTitle("DSA Viewer - Registration & Subtraction Only")
        self.resize(1600, 900)

        # Drag & drop
        self.setAcceptDrops(True)

        # Split views
        splitter = QSplitter(Qt.Horizontal)

        self.viewer1 = DicomViewer(dsa_mode=False)
        splitter.addWidget(self.viewer1)

        self.viewer2 = DicomViewer(dsa_mode=True)
        splitter.addWidget(self.viewer2)

        splitter.setSizes([800, 800])
        self.setCentralWidget(splitter)

        # Toolbar
        self.setup_toolbar()

        # Cine timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Status bar
        self.setup_statusbar()

        self.frames = None
        self.frame_index = 0
        self.play_speed = 100
        self.dicom_dataset = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith('.dcm'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
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
            self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if filename:
            self.load_dicom_file(filename)

    def load_dicom_file(self, filename):
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

            short_name = os.path.basename(filename)
            self.viewer1.set_frames(frames, short_name, original_dicom=ds)
            self.viewer2.set_frames(frames, short_name, original_dicom=ds)

            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self.update_frame_label()

            self.statusBar().showMessage(f"Loaded: {short_name} ({len(frames)} frames)", 3000)

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
        print("=" * 70)
        print("DSA VIEWER - Histogram Matching GPU Enabled")
        print("=" * 70)
        get_gpu_memory_info()
    else:
        print("=" * 70)
        print("DSA VIEWER - CPU Mode")
        print(" Install CuPy for GPU-accelerated histogram matching")
        print("=" * 70)

    window = MainWindow()
    window.show()

    app.exec()

    if GPU_AVAILABLE:
        clear_gpu_cache()
