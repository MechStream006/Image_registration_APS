import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from math import log10

# =========================================================
# ------------------- CONFIGURATION SECTION -------------------
# =========================================================

# INPUT/OUTPUT PATHS
# Can be a file or directory - if directory, will process all .dcm files
input_dicom_path = r"D:\LAVANYA\1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
output_main_folder = r"C:\Users\VI3CATH ADMIN\Desktop\test dicom loop\combined_pipeline_results"

# PREPROCESSING (Doc 1) SETTINGS
GAMMA = 0.5
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)
WIENER_KERNEL = 5
ENABLE_PREVIEW = False
ENABLE_PREVIEW = False

# ROI SETTINGS FOR METRICS
ROI_COORDS = [
    (536, 586, 536, 586),
    (515, 565, 725, 775),
    (640, 690, 440, 490),
    (825, 875, 630, 680)
]
ROI_COORDS = [(x1, x1+50, y1, y1+50) for (x1, x2, y1, y2) in ROI_COORDS]

# DSA PROCESSING (Doc 2) SETTINGS
USE_HISTOGRAM_MATCHING = True
HIST_MATCH_BINS = 256
MASK_FRAME_IDX = 0

# 16-BIT SPECIFIC SETTINGS
DTYPE_PRECISION = cp.uint16
DTYPE_COMPUTE = cp.float32
DICOM_PIXEL_MAX = 65535.0

# GPU MEMORY
GPU_ENABLED = True
MAX_WORKERS = 4

os.makedirs(output_main_folder, exist_ok=True)

# =========================================================
# ------------------- GPU MEMORY MANAGEMENT -------------------
# =========================================================

def get_gpu_memory_info():
    """Check available GPU memory."""
    if not GPU_ENABLED:
        return
    mempool = cp.get_default_memory_pool()
    print(f"GPU Memory Used: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"GPU Memory Total: {mempool.total_bytes() / 1e9:.2f} GB")


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if not GPU_ENABLED:
        return
    cp.get_default_memory_pool().free_all_blocks()


# =========================================================
# ------------------- FLOAT CONVERSION UTILITIES -------------------
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
# ------------------- IMAGE PROCESSING (PREPROCESSING) -------------------
# =========================================================

def gamma_correction(img_float, gamma):
    """Apply gamma correction."""
    return np.clip(np.power(img_float, gamma), 0, 1)


def apply_clahe(img_float, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    img_uint16 = np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_uint16)
    return np.clip(clahe_img.astype(np.float32) / 65535, 0, 1)


def wiener_filter(img_float, kernel_size=5):
    """Apply Wiener filter for noise reduction."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    local_mean = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_mean_sq = cv2.filter2D(img_float**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-10)
    noise_var = np.mean(local_var) * 0.01
    gain = np.clip((local_var - noise_var) / local_var, 0, 1)
    result = local_mean + gain * (img_float - local_mean)
    return np.clip(result, 0, 1)


def apply_windowing(img_float, WL=0.4, WW=0.6):
    """Apply windowing for intensity adjustment."""
    img_min = WL - WW / 2
    img_max = WL + WW / 2
    windowed = np.clip(img_float, img_min, img_max)
    return np.clip((windowed - img_min) / (img_max - img_min), 0, 1)


def apply_unsharp_mask(img_float, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """Apply unsharp masking for sharpening."""
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    sharpened = (amount + 1) * img_float - amount * blurred
    return np.clip(sharpened, 0, 1)


# =========================================================
# ------------------- PREPROCESSING PIPELINE -------------------
# =========================================================

def process_frame(frame, gamma, clahe_clip=2.0, clahe_tiles=(8, 8), wiener_kernel=5, preview=False):
    """
    Apply complete preprocessing pipeline:
    Wiener → Gamma → CLAHE → Unsharp Masking → Windowing
    """
    frame_float = to_float(frame)
    
    # Sequence: IN --> Wiener --> Gamma --> CLAHE --> Unsharp Masking --> Windowing
    wi = wiener_filter(frame_float, kernel_size=wiener_kernel)
    gm = gamma_correction(wi, gamma=gamma)
    cl = apply_clahe(gm, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    unsharp = apply_unsharp_mask(cl)
    win = apply_windowing(unsharp)
    
    if preview:
        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(frame_float, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
        axs[1].imshow(wi, cmap='gray'); axs[1].set_title("Wiener"); axs[1].axis('off')
        axs[2].imshow(gm, cmap='gray'); axs[2].set_title("Gamma"); axs[2].axis('off')
        axs[3].imshow(cl, cmap='gray'); axs[3].set_title("CLAHE"); axs[3].axis('off')
        axs[4].imshow(unsharp, cmap='gray'); axs[4].set_title("Unsharp"); axs[4].axis('off')
        axs[5].imshow(win, cmap='gray'); axs[5].set_title("Windowing"); axs[5].axis('off')
        
        for ax in axs:
            for (x1, x2, y1, y2) in ROI_COORDS:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'original_float': frame_float,
        'wiener': wi,
        'gamma': gm,
        'clahe': cl,
        'unsharp_masked': unsharp,
        'windowed': win
    }


# =========================================================
# ------------------- ROI & METRICS (PREPROCESSING) -------------------
# =========================================================

def calculate_roi_metrics(original, processed):
    """Calculate SNR and CNR metrics for ROI."""
    orig, proc = original.astype(np.float32), processed.astype(np.float32)
    snr = np.mean(proc) / (np.std(proc) + 1e-8)
    cnr = np.abs(np.mean(proc) - np.mean(orig)) / (np.std(orig) + 1e-8)
    return {'SNR': snr, 'CNR': cnr}


def extract_rois_from_frame(frame, rois):
    """Extract ROIs from frame."""
    return [frame[y1:y2, x1:x2] for (x1, x2, y1, y2) in rois]


def visualize_rois(frame, rois, ax, title="Frame with ROIs"):
    """Visualize ROIs on frame."""
    ax.imshow(frame, cmap="gray")
    for (x1, x2, y1, y2) in rois:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')


# =========================================================
# ------------------- HISTOGRAM MATCHING (GPU) -------------------
# =========================================================

def histogram_matching_gpu(source_gpu: cp.ndarray, reference_gpu: cp.ndarray, bins: int = 256) -> cp.ndarray:
    """
    Match histogram of source to reference on GPU.
    Optimized for 16-bit data with proper dtype handling.
    """
    source = source_gpu.astype(cp.float32, copy=False)
    reference = reference_gpu.astype(cp.float32, copy=False)
    
    src_min, src_max = float(source.min()), float(source.max())
    ref_min, ref_max = float(reference.min()), float(reference.max())
    
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


# =========================================================
# ------------------- PHASE CORRELATION (GPU) -------------------
# =========================================================

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


def phase_correlation(mask_gpu: cp.ndarray, live_gpu: cp.ndarray):
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


def apply_subpixel_shift(img_gpu: cp.ndarray, dy: float, dx: float) -> cp.ndarray:
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


# =========================================================
# ------------------- METRICS (DSA) -------------------
# =========================================================

def mse(img1, img2):
    """Mean squared error (GPU-compatible)."""
    return float(cp.mean((img1 - img2) ** 2))


def psnr(img1, img2):
    """Peak signal-to-noise ratio for 16-bit data."""
    mse_val = mse(img1, img2)
    if mse_val < 1e-10:
        return float("inf")
    return 20 * log10(DICOM_PIXEL_MAX / np.sqrt(mse_val))


def mutual_information_gpu(hgram_np):
    """Mutual Information from 2D histogram."""
    pxy = hgram_np / float(np.sum(hgram_np) + 1e-10)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py) + 1e-10
    nzs = pxy > 0
    return float(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))


def compute_metrics_gpu(mask_gpu, aligned_gpu):
    """Compute all metrics on GPU."""
    mask_f = mask_gpu.astype(cp.float32, copy=False)
    aligned_f = aligned_gpu.astype(cp.float32, copy=False)
    
    m_mse = mse(mask_f, aligned_f)
    m_psnr = psnr(mask_f, aligned_f)
    
    mask_mean = mask_f.mean()
    aligned_mean = aligned_f.mean()
    ncc_num = cp.sum((mask_f - mask_mean) * (aligned_f - aligned_mean))
    ncc_den = cp.sqrt(cp.sum((mask_f - mask_mean) ** 2) * 
                      cp.sum((aligned_f - aligned_mean) ** 2) + 1e-10)
    m_ncc = float(ncc_num / ncc_den)
    
    mask_np = cp.asnumpy(mask_f).astype(np.uint16)
    aligned_np = cp.asnumpy(aligned_f).astype(np.uint16)
    m_ssim = ssim(mask_np, aligned_np, data_range=DICOM_PIXEL_MAX)
    
    hgram, _, _ = np.histogram2d(mask_np.ravel(), aligned_np.ravel(), bins=256, 
                                  range=[[0, DICOM_PIXEL_MAX], [0, DICOM_PIXEL_MAX]])
    m_mi = mutual_information_gpu(hgram)
    
    return m_mse, m_psnr, m_ncc, m_ssim, m_mi


# =========================================================
# ------------------- SAVE UTILITIES -------------------
# =========================================================

def save_metrics_to_excel(metrics, out_path, auto_rename=True):
    """Save metrics to Excel with auto-rename if file exists."""
    if os.path.exists(out_path) and auto_rename:
        base, ext = os.path.splitext(out_path)
        counter = 1
        new_path = f"{base}_v{counter}{ext}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base}_v{counter}{ext}"
        out_path = new_path
        print(f"⚠️ File exists, saving instead as: {out_path}")
    
    pd.DataFrame(metrics).to_excel(out_path, index=False, engine="openpyxl")
    print(f"✅ Saved metrics Excel: {out_path}")


def save_histograms_with_charts(histogram_data, out_path):
    """Save histogram data to Excel."""
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet("Histograms")
    worksheet.write_row(0, 0, ["Frame", "Filter", "Bin", "Frequency"])
    
    for i, row in enumerate(histogram_data, start=1):
        worksheet.write_row(i, 0, [row['Frame'], row['Filter'], row['Bin'], row['Frequency']])
    
    workbook.close()
    print(f"✅ Saved histogram Excel: {out_path}")


def save_multiframe_dicom(original_dcm, frames_float, out_path, description_suffix=""):
    """Save preprocessed frames as multi-frame DICOM."""
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
        new_dcm.SeriesDescription = f"{getattr(new_dcm, 'SeriesDescription', 'Unnamed')} {description_suffix}"
    
    new_dcm.PixelData = np.stack(frames_scaled).tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    
    print(f"✅ Saved multi-frame DICOM: {out_path}")


# =========================================================
# ------------------- COMBINED PIPELINE -------------------
# =========================================================

def process_combined_pipeline(dicom_path, gamma, clahe_clip, clahe_tiles, wiener_kernel,
                              preview=False):
    """
    Complete pipeline:
    1. Load DICOM
    2. Preprocess frames (Doc 1)
    3. Align & Subtract (Doc 2)
    4. Save results
    """
    
    print(f"\n{'='*70}")
    print(f"STAGE 1: LOADING DICOM")
    print(f"{'='*70}")
    
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    
    print(f"✓ Loaded DICOM with {arr.shape[0]} frames")
    print(f"✓ Frame dimensions: {arr.shape[1:]} | Data type: {arr.dtype}")
    print(f"✓ Pixel value range: [{arr.min()}, {arr.max()}]")
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(dicom_path))[0]
    output_dir = os.path.join(output_main_folder, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================
    # STAGE 1: PREPROCESSING (Doc 1)
    # =====================================================
    
    print(f"\n{'='*70}")
    print(f"STAGE 1: PREPROCESSING FRAMES")
    print(f"{'='*70}")
    
    preprocessed_frames = []
    preprocessing_metrics = []
    histogram_data = []
    
    for i, frame in enumerate(arr):
        print(f"🔄 Preprocessing frame {i}/{arr.shape[0]-1}...")
        
        res = process_frame(frame, gamma, clahe_clip, clahe_tiles, wiener_kernel, preview=preview)
        preprocessed_frames.append(res['windowed'])
        
        # Extract ROI metrics
        for j, (x1, x2, y1, y2) in enumerate(ROI_COORDS):
            orig_roi = res['clahe'][y1:y2, x1:x2]
            proc_roi = res['windowed'][y1:y2, x1:x2]
            
            roi_metric = calculate_roi_metrics(orig_roi, proc_roi)
            roi_metric.update({'Frame': i, 'ROI': j})
            preprocessing_metrics.append(roi_metric)
        
        # Histogram data
        for ftype, img in zip(['CLAHE', 'Windowed'], [res['clahe'], res['windowed']]):
            hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
            for b, freq in enumerate(hist):
                histogram_data.append({'Frame': i, 'Filter': ftype, 'Bin': b, 'Frequency': freq})
    
    # Save preprocessed DICOM
    preprocessed_dicom_path = os.path.join(output_dir, f"{base_name}_preprocessed.dcm")
    save_multiframe_dicom(ds, preprocessed_frames, preprocessed_dicom_path, "(Preprocessed)")
    
    # Save preprocessing metrics
    preprocessing_metrics_path = os.path.join(output_dir, f"{base_name}_preprocessing_metrics.xlsx")
    save_metrics_to_excel(preprocessing_metrics, preprocessing_metrics_path)
    
    histogram_excel_path = os.path.join(output_dir, f"{base_name}_histograms.xlsx")
    save_histograms_with_charts(histogram_data, histogram_excel_path)
    
    print(f"✅ Preprocessing complete. Saved preprocessed DICOM and metrics.")
    
    # =====================================================
    # STAGE 2: DSA (Doc 2) - ALIGNMENT & SUBTRACTION
    # =====================================================
    
    print(f"\n{'='*70}")
    print(f"STAGE 2: DSA - ALIGNMENT & SUBTRACTION")
    print(f"{'='*70}")
    
    # Convert preprocessed frames to float32 for GPU
    frames_cpu = np.array([from_float(f) for f in preprocessed_frames]).astype(np.float32)
    
    mask_gpu = cp.asarray(frames_cpu[MASK_FRAME_IDX])
    print(f"✓ Mask frame (index {MASK_FRAME_IDX}) loaded to GPU. Shape: {mask_gpu.shape}")
    
    dsa_metrics_list = []
    frames_folder = os.path.join(output_dir, f"{base_name}_frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    def process_dsa_frame(idx):
        """Process single frame for DSA on GPU."""
        live_gpu = cp.asarray(frames_cpu[idx])
        
        # Phase correlation alignment
        dy, dx, corr_gpu = phase_correlation(mask_gpu, live_gpu)
        
        shift_magnitude = (dy**2 + dx**2)**0.5
        if shift_magnitude < 0.1:
            print(f"  ⚠️  Frame {idx}: Very small shift ({dy:.4f}, {dx:.4f}) px")
        elif shift_magnitude > 50:
            print(f"  ⚠️  Frame {idx}: Large shift ({dy:.2f}, {dx:.2f}) px")
        
        live_aligned_gpu = apply_subpixel_shift(live_gpu, dy, dx)
        
        # Histogram matching
        if USE_HISTOGRAM_MATCHING:
            live_hist_matched_gpu = histogram_matching_gpu(live_aligned_gpu, mask_gpu, bins=HIST_MATCH_BINS)
        else:
            live_hist_matched_gpu = live_aligned_gpu
        
        # Get numpy copies
        mask_np = cp.asnumpy(mask_gpu).astype(np.uint16)
        aligned_np = cp.asnumpy(live_aligned_gpu).astype(np.uint16)
        hist_matched_np = cp.asnumpy(live_hist_matched_gpu).astype(np.uint16)
        live_np = cp.asnumpy(live_gpu).astype(np.uint16)
        
        # Subtraction
        sub_gpu = live_hist_matched_gpu - mask_gpu
        sub_min = float(sub_gpu.min())
        sub_max = float(sub_gpu.max())
        
        if sub_max > sub_min:
            sub_normalized = ((sub_gpu - sub_min) / (sub_max - sub_min)) * DICOM_PIXEL_MAX
        else:
            sub_normalized = cp.zeros_like(sub_gpu)
        
        sub_normalized = cp.clip(sub_normalized, 0, DICOM_PIXEL_MAX)
        sub_np = cp.asnumpy(sub_normalized).astype(np.uint16)
        
        # CLAHE enhancement for visualization
        sub_enhanced = exposure.equalize_adapthist(
            sub_np.astype(np.float32) / DICOM_PIXEL_MAX, 
            clip_limit=0.03
        )
        sub_enhanced = (sub_enhanced * DICOM_PIXEL_MAX).astype(np.uint16)
        
        # Compute metrics
        m_mse_before, m_psnr_before, m_ncc_before, m_ssim_before, m_mi_before = compute_metrics_gpu(
            mask_gpu, live_aligned_gpu)
        m_mse_after, m_psnr_after, m_ncc_after, m_ssim_after, m_mi_after = compute_metrics_gpu(
            mask_gpu, live_hist_matched_gpu)
        
        dsa_metrics_list.append({
            "Frame": idx,
            "dy": dy, "dx": dx,
            "MSE_BeforeHistMatch": m_mse_before,
            "PSNR_BeforeHistMatch": m_psnr_before,
            "NCC_BeforeHistMatch": m_ncc_before,
            "SSIM_BeforeHistMatch": m_ssim_before,
            "MI_BeforeHistMatch": m_mi_before,
            "MSE_AfterHistMatch": m_mse_after,
            "PSNR_AfterHistMatch": m_psnr_after,
            "NCC_AfterHistMatch": m_ncc_after,
            "SSIM_AfterHistMatch": m_ssim_after,
            "MI_AfterHistMatch": m_mi_after,
            "Sub_Min": sub_min,
            "Sub_Max": sub_max,
            "HistMatch_Enabled": USE_HISTOGRAM_MATCHING
        })
        
        # Visualization: Correlation surface
        try:
            corr_np = cp.asnumpy(corr_gpu)
            corr_vis = np.fft.fftshift(corr_np)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_title(f"Correlation Surface - Frame {idx}")
            im = ax.imshow(corr_vis, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            plt.savefig(os.path.join(frames_folder, f"correlation_surface_frame_{idx:03d}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Correlation plot failed for frame {idx}: {e}")
        
        # Visualization: Comparison grid
        try:
            fig = plt.figure(figsize=(18, 18))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Row 1: Original frames
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(mask_np, cmap='gray', vmin=0, vmax=DICOM_PIXEL_MAX)
            ax1.set_title("Mask (Reference)")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(live_np, cmap='gray', vmin=0, vmax=DICOM_PIXEL_MAX)
            ax2.set_title(f"Live Frame {idx} (Original)")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(aligned_np, cmap='gray', vmin=0, vmax=DICOM_PIXEL_MAX)
            ax3.set_title(f"After Alignment\nShift: ({dy:.2f}, {dx:.2f}) px")
            ax3.axis('off')
            
            # Row 2: Histogram matching
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.hist(mask_np.ravel(), bins=256, alpha=0.7, color='blue', label='Mask', range=(0, DICOM_PIXEL_MAX))
            ax4.hist(aligned_np.ravel(), bins=256, alpha=0.7, color='red', label='Aligned', range=(0, DICOM_PIXEL_MAX))
            ax4.set_title("Histograms: Before Match")
            ax4.legend()
            ax4.set_ylabel("Frequency")
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(hist_matched_np, cmap='gray', vmin=0, vmax=DICOM_PIXEL_MAX)
            status = "Enabled" if USE_HISTOGRAM_MATCHING else "Disabled"
            ax5.set_title(f"After Histogram Match ({status})")
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.hist(mask_np.ravel(), bins=256, alpha=0.7, color='blue', label='Mask', range=(0, DICOM_PIXEL_MAX))
            ax6.hist(hist_matched_np.ravel(), bins=256, alpha=0.7, color='green', label='Matched', range=(0, DICOM_PIXEL_MAX))
            ax6.set_title("Histograms: After Match")
            ax6.legend()
            ax6.set_ylabel("Frequency")
            
            # Row 3: Subtraction results
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.imshow(sub_np, cmap='gray')
            ax7.set_title(f"Subtracted (Normalized)\nOriginal range: [{sub_min:.0f}, {sub_max:.0f}]")
            ax7.axis('off')
            
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.imshow(sub_enhanced, cmap='gray')
            improvement = ((m_mse_before - m_mse_after) / (m_mse_before + 1e-10) * 100)
            ax8.set_title(f"CLAHE Enhanced\nMSE improvement: {improvement:.1f}%")
            ax8.axis('off')
            
            ax9 = fig.add_subplot(gs[2, 2])
            diff_raw = cp.asnumpy(sub_gpu)
            im = ax9.imshow(diff_raw, cmap='RdBu_r', 
                           vmin=-np.percentile(np.abs(diff_raw), 99),
                           vmax=np.percentile(np.abs(diff_raw), 99))
            ax9.set_title("Signed Difference Map")
            ax9.axis('off')
            plt.colorbar(im, ax=ax9, fraction=0.046)
            
            plt.savefig(os.path.join(frames_folder, f"comparison_frame_{idx:03d}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Comparison visualization failed for frame {idx}: {e}")
        
        return sub_np
    
    # Process all frames in parallel (excluding mask frame)
    frame_indices = [i for i in range(arr.shape[0]) if i != MASK_FRAME_IDX]
    print(f"Processing {len(frame_indices)} frames in parallel...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        subtracted_frames = list(executor.map(process_dsa_frame, frame_indices))
    
    # =====================================================
    # SAVE DSA RESULTS
    # =====================================================
    
    print(f"\n{'='*70}")
    print(f"STAGE 3: SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save combined multi-frame DICOM (subtracted frames)
    combined_ds = ds.copy()
    combined_array = np.stack(subtracted_frames, axis=0)
    combined_ds.PixelData = combined_array.astype(np.uint16).tobytes()
    combined_ds.NumberOfFrames = combined_array.shape[0]
    combined_ds.BitsAllocated = 16
    combined_ds.BitsStored = 16
    combined_ds.HighBit = 15
    combined_ds.SeriesInstanceUID = generate_uid()
    combined_ds.SOPInstanceUID = generate_uid()
    hist_status = "with HistMatch" if USE_HISTOGRAM_MATCHING else "no HistMatch"
    combined_ds.SeriesDescription = f"DSA Subtracted ({hist_status}) - {getattr(ds, 'SeriesDescription', 'Unknown')}"
    
    combined_filename = f"{base_name}_dsa_subtracted.dcm"
    combined_path = os.path.join(output_dir, combined_filename)
    combined_ds.save_as(combined_path)
    print(f"✓ Saved DSA subtracted DICOM: {combined_filename}")
    
    # Save DSA metrics
    dsa_metrics_path = os.path.join(output_dir, f"{base_name}_dsa_metrics.xlsx")
    save_metrics_to_excel(dsa_metrics_list, dsa_metrics_path)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Complete pipeline finished successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Outputs generated:")
    print(f"   - Preprocessed DICOM: {base_name}_preprocessed.dcm")
    print(f"   - DSA Subtracted DICOM: {combined_filename}")
    print(f"   - Preprocessing metrics: {base_name}_preprocessing_metrics.xlsx")
    print(f"   - Histogram data: {base_name}_histograms.xlsx")
    print(f"   - DSA metrics: {base_name}_dsa_metrics.xlsx")
    print(f"   - Visualization frames: {base_name}_frames/")
    
    if USE_HISTOGRAM_MATCHING:
        avg_improvement = np.mean([m['MSE_BeforeHistMatch'] - m['MSE_AfterHistMatch'] for m in dsa_metrics_list])
        avg_psnr_gain = np.mean([m['PSNR_AfterHistMatch'] - m['PSNR_BeforeHistMatch'] for m in dsa_metrics_list])
        print(f"\n📊 DSA Improvements (with Histogram Matching):")
        print(f"   - Average MSE improvement: {avg_improvement:.2f}")
        print(f"   - Average PSNR gain: {avg_psnr_gain:.2f} dB")
    
    clear_gpu_cache()
    print(f"\n✅ All processing complete.\n")


# =========================================================
# ------------------- MAIN -------------------
# =========================================================

if __name__ == "__main__":
    if GPU_ENABLED:
        print("🖥️  GPU Acceleration: ENABLED")
        get_gpu_memory_info()
    else:
        print("🖥️  GPU Acceleration: DISABLED")
    
    print(f"\n{'='*70}")
    print(f"COMBINED DICOM PREPROCESSING & DSA PIPELINE")
    print(f"{'='*70}\n")
    
    # Check if input_dicom_path is a file or directory
    if os.path.isfile(input_dicom_path):
        # Single file
        print(f"Processing single file: {input_dicom_path}\n")
        try:
            process_combined_pipeline(
                input_dicom_path,
                GAMMA,
                CLAHE_CLIP,
                CLAHE_TILES,
                WIENER_KERNEL,
                preview=ENABLE_PREVIEW
            )
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    elif os.path.isdir(input_dicom_path):
        # Directory - process all DICOM files
        print(f"Processing directory: {input_dicom_path}\n")
        dicom_files = [f for f in os.listdir(input_dicom_path) if f.lower().endswith('.dcm')]
        
        if not dicom_files:
            print(f"⚠️  No DICOM files found in {input_dicom_path}")
        else:
            print(f"Found {len(dicom_files)} DICOM files to process\n")
            
            for file_name in dicom_files:
                file_path = os.path.join(input_dicom_path, file_name)
                print(f"\n{'#'*70}")
                print(f"Processing: {file_name}")
                print(f"{'#'*70}")
                
                try:
                    process_combined_pipeline(
                        file_path,
                        GAMMA,
                        CLAHE_CLIP,
                        CLAHE_TILES,
                        WIENER_KERNEL,
                        preview=ENABLE_PREVIEW
                    )
                except Exception as e:
                    print(f"❌ Failed to process {file_name}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print(f"❌ Input path does not exist or is not accessible: {input_dicom_path}")
    
    if GPU_ENABLED:
        get_gpu_memory_info()