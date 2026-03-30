import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from pydicom.dataset import FileDataset
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

# =========================================================
# ----------------- Float Conversion Utilities -----------------
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
# ----------------- ROI Utilities -----------------
# =========================================================
def extract_rois_from_frame(frame, rois):
    return [frame[y1:y2, x1:x2] for (x1, x2, y1, y2) in rois]

def visualize_rois(frame, rois, ax, title="Frame with ROIs"):
    ax.imshow(frame, cmap="gray")
    for (x1, x2, y1, y2) in rois:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')

# =========================================================
# ----------------- Image Processing (float-safe) -----------------
# =========================================================
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

def apply_windowing(img_float, WL=0.4, WW=0.6):
    img_min = WL - WW / 2
    img_max = WL + WW / 2
    windowed = np.clip(img_float, img_min, img_max)
    return np.clip((windowed - img_min) / (img_max - img_min), 0, 1)

def apply_unsharp_mask(img_float, kernel_size=(5,5), sigma=1.0, amount=1.5):
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    sharpened = (amount + 1) * img_float - amount * blurred
    return np.clip(sharpened, 0, 1)

def process_frame(frame, gamma, clahe_clip=2.0, clahe_tiles=(8, 8), wiener_kernel=5, preview=False):
    frame_float = to_float(frame)

    # Sequence: IN --> Wiener --> Gamma --> CLAHE --> Unsharp Masking --> Windowing
    wi = wiener_filter(frame_float, kernel_size=wiener_kernel)
    gm = gamma_correction(wi, gamma=gamma)
    cl = apply_clahe(gm, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    unsharp = apply_unsharp_mask(cl)
    win = apply_windowing(unsharp)

    if preview:
        rois = [(536, 586, 536, 586),(515, 565, 725, 775),
                (640, 690, 440, 490),(825, 875, 630, 680)]
        rois = [(x1, x1+50, y1, y1+50) for (x1, x2, y1, y2) in rois]

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(frame_float, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
        axs[1].imshow(wi, cmap='gray'); axs[1].set_title("Wiener"); axs[1].axis('off')
        axs[2].imshow(gm, cmap='gray'); axs[2].set_title("Gamma"); axs[2].axis('off')
        axs[3].imshow(cl, cmap='gray'); axs[3].set_title("CLAHE"); axs[3].axis('off')
        axs[4].imshow(unsharp, cmap='gray'); axs[4].set_title("Unsharp"); axs[4].axis('off')
        axs[5].imshow(win, cmap='gray'); axs[5].set_title("Windowing"); axs[5].axis('off')
        for ax in axs:
            for (x1, x2, y1, y2) in rois:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
        plt.tight_layout(); plt.show()

    return {
        'original_float': frame_float,
        'wiener': wi,
        'gamma': gm,
        'clahe': cl,
        'unsharp_masked': unsharp,
        'windowed': win
    }

def perform_dsa(mask_frame, contrast_frame, normalization='standard'):
    """
    Perform Digital Subtraction Angiography
    
    Args:
        mask_frame: Pre-contrast image (float, 0-1)
        contrast_frame: Post-contrast image (float, 0-1)
        normalization: 'standard', 'weighted', or 'log'
    
    Returns:
        dsa_result: DSA subtraction image
    """
    if normalization == 'standard':
        # Simple subtraction
        dsa = contrast_frame - mask_frame
        
    elif normalization == 'weighted':
        # Weighted subtraction (reduces noise)
        epsilon = 1e-6
        dsa = (contrast_frame - mask_frame) / (mask_frame + epsilon)
        
    elif normalization == 'log':
        # Logarithmic subtraction (better for varying intensities)
        epsilon = 1e-6
        dsa = np.log(mask_frame + epsilon) - np.log(contrast_frame + epsilon)
    
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    # Normalize to [0, 1] range
    dsa_min, dsa_max = dsa.min(), dsa.max()
    if dsa_max > dsa_min:
        dsa_normalized = (dsa - dsa_min) / (dsa_max - dsa_min)
    else:
        dsa_normalized = np.zeros_like(dsa)
    
    return np.clip(dsa_normalized, 0, 1)

def save_multiframe_dicom(original_dcm, frames_float, out_path):
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
 
    new_dcm.PixelData = np.stack(frames_scaled).tobytes()
    new_dcm.save_as(out_path, write_like_original=False)
    print(f"✅ Saved multi-frame DICOM (16-bit): {out_path}")

def calculate_metrics(original, processed):
    orig, proc = original.astype(np.float32), processed.astype(np.float32)
    snr = np.mean(proc) / (np.std(proc) + 1e-8)
    cnr = np.abs(np.mean(proc) - np.mean(orig)) / (np.std(orig) + 1e-8)
    return {'SNR': snr, 'CNR': cnr}

def process_dicom_with_dsa(dicom_path, gamma, clahe_clip, clahe_tiles,
                           wiener_kernel, mask_frame_index,
                           dsa_normalization='standard',
                           out_dsa_dcm=None,
                           excel_metrics_path=None,
                           histogram_excel_path=None,
                           preview=True):
    """
    Complete pipeline with preprocessing and DSA (no registration)
    
    Args:
        mask_frame_index: Index of the frame to use as mask (pre-contrast)
        dsa_normalization: 'standard', 'weighted', or 'log'
    """
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    if arr.ndim == 2: arr = arr[np.newaxis, :, :]

    frames_preprocessed = []
    frames_dsa = []
    metrics = []
    histogram_data = []

    rois = [
        (536, 586, 536, 586),
        (515, 565, 725, 775),
        (640, 690, 440, 490),
        (825, 875, 630, 680)
    ]
    rois = [(x1, x1+50, y1, y1+50) for (x1, x2, y1, y2) in rois]

    # Step 1: Preprocess all frames
    print("\n📊 Step 1: Preprocessing frames...")
    for i, frame in enumerate(arr):
        print(f"  Processing frame {i}")
        res = process_frame(frame, gamma, clahe_clip, clahe_tiles, wiener_kernel, preview=(preview and i==0))
        frames_preprocessed.append(res['windowed'])

    # Step 2: Get mask frame
    print(f"\n🎭 Step 2: Using frame {mask_frame_index} as mask (pre-contrast)")
    mask_frame = frames_preprocessed[mask_frame_index]

    # Step 3: Perform DSA directly on preprocessed frames (no registration)
    print(f"\n👉 Step 3: Performing DSA (method: {dsa_normalization})...")
    for i, contrast_frame in enumerate(frames_preprocessed):
        if i == mask_frame_index:
            # Mask frame produces zero DSA
            dsa_result = np.zeros_like(mask_frame)
            print(f"  Frame {i}: Mask frame (zero DSA)")
        else:
            dsa_result = perform_dsa(mask_frame, contrast_frame, normalization=dsa_normalization)
            print(f"  Frame {i}: DSA computed")
        
        frames_dsa.append(dsa_result)

        # Calculate metrics for ROIs
        for j, (x1, x2, y1, y2) in enumerate(rois):
            orig_roi = frames_preprocessed[i][y1:y2, x1:x2]
            dsa_roi = dsa_result[y1:y2, x1:x2]
            roi_metric = calculate_metrics(orig_roi, dsa_roi)
            roi_metric.update({
                'Frame': i, 
                'ROI': j
            })
            metrics.append(roi_metric)

        # Histogram data
        for ftype, img in zip(['Preprocessed', 'DSA'], 
                             [frames_preprocessed[i], dsa_result]):
            hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
            for b, freq in enumerate(hist):
                histogram_data.append({
                    'Frame': i, 
                    'Filter': ftype, 
                    'Bin': b, 
                    'Frequency': freq
                })

    # Save DSA DICOM
    if out_dsa_dcm:
        save_multiframe_dicom(ds, frames_dsa, out_dsa_dcm)

    # Save metrics to Excel
    if excel_metrics_path and metrics:
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_excel(excel_metrics_path, index=False, sheet_name='Metrics')
        print(f"✅ Saved metrics: {excel_metrics_path}")

    # Save histogram data to Excel
    if histogram_excel_path and histogram_data:
        df_hist = pd.DataFrame(histogram_data)
        df_hist.to_excel(histogram_excel_path, index=False, sheet_name='Histograms')
        print(f"✅ Saved histograms: {histogram_excel_path}")

    return frames_preprocessed, frames_dsa, metrics, histogram_data


if __name__ == "__main__":
    dicom_path = r"D:\Rohith\RAW_AUTOPIXEL\Neuro_RAW/1.2.826.0.1.3680043.2.1330.2640165.2408301127330001.5.443_Raw_anon.dcm"
    
    # Output paths
    output_dsa_dcm = r"D:/Rohith/Auto_pixel_shift/APS/output_2/dsa_results.dcm"
    metrics_excel_path = r"D:/Rohith/Auto_pixel_shift/APS/output_2/dsa_metrics.xlsx"
    histogram_excel_path = r"D:/Rohith/Auto_pixel_shift/APS/output_2/dsa_histograms.xlsx"

    # Parameters
    gamma = 0.5
    clahe_clip = 2.0
    clahe_tiles = (8, 8)
    wiener_kernel = 5
    mask_frame_index = 0  # Frame to use as mask (pre-contrast)
    dsa_normalization = 'standard'  # 'standard', 'weighted', or 'log'
    preview = True

    # Run complete pipeline
    preprocessed, dsa_frames, metrics, histogram_data = process_dicom_with_dsa(
        dicom_path=dicom_path,
        gamma=gamma,
        clahe_clip=clahe_clip,
        clahe_tiles=clahe_tiles,
        wiener_kernel=wiener_kernel,
        mask_frame_index=mask_frame_index,
        dsa_normalization=dsa_normalization,
        out_dsa_dcm=output_dsa_dcm,
        excel_metrics_path=metrics_excel_path,
        histogram_excel_path=histogram_excel_path,
        preview=preview
    )
    
    print("\n✅ Processing complete!")