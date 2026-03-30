import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
 
# ------------------- Conversion Utilities -------------------
def to_float(img_uint16):
    return img_uint16.astype(np.float32) / 65535.0
 
def from_float(img_float):
    return np.clip(img_float * 65535, 0, 65535).astype(np.uint16)
 
# ------------------- Enhancement Functions -------------------
def gamma_correction(img_float, gamma=0.95):
    return np.clip(np.power(img_float, gamma), 0, 1)
 
def gaussian_denoise(img_float, sigma=0.6):
    return cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)
 
def nlm_denoise(img_float, h=0.0125):
    img_8u = np.uint8(np.clip(img_float * 255, 0, 255))
    denoised = cv2.fastNlMeansDenoising(img_8u, None, h=h*255, templateWindowSize=7, searchWindowSize=21)
    return denoised.astype(np.float32) / 255.0
 
def clahe_enhance(img_float, clip_limit=2, tile_grid_size=(4,4)):
    img_8u = np.uint8(img_float * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_8u)
    return enhanced.astype(np.float32) / 255.0
 
def unsharp_mask(img_float, amount=0.45, radius=0.35):
    blurred = cv2.GaussianBlur(img_float, (0,0), radius)
    sharpened = cv2.addWeighted(img_float, 1+amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)
 
# ------------------- ROI & Metrics -------------------
def extract_rois_from_frame(frame, rois):
    return [frame[y1:y2, x1:x2] for (x1, x2, y1, y2) in rois]
 
def calc_snr(roi):
    return np.mean(roi) / (np.std(roi) + 1e-8)
 
def calc_cnr(signal_roi, background_roi):
    return abs(np.mean(signal_roi) - np.mean(background_roi)) / (np.std(background_roi) + 1e-8)
 
def calc_entropy(roi):
    return shannon_entropy(roi)
 
# ------------------- Main Pipeline -------------------
def process_dicom_with_metrics(input_path, output_path, excel_path, rois, visualize=True):
    ds = pydicom.dcmread(input_path)
    orig_frames = ds.pixel_array
    if orig_frames.ndim == 2:
        orig_frames = orig_frames[np.newaxis, :, :]
 
    processed_frames = []
    metrics = []
 
    print(f"Processing {orig_frames.shape[0]} frames...")
 
    for i, frame in enumerate(orig_frames):
        img = to_float(frame)
 
        # ---- Pipeline ----
        img = gamma_correction(img, gamma=0.5)
        img = gaussian_denoise(img, sigma=0.6)
        img = nlm_denoise(img, h=0.0125)
        img = clahe_enhance(img, clip_limit=2, tile_grid_size=(4,4))
        img = unsharp_mask(img, amount=0.45, radius=0.35)
 
        proc_uint16 = from_float(img)
        processed_frames.append(proc_uint16)
 
        # ---- Metrics ----
        orig_f = to_float(frame)
        proc_f = to_float(proc_uint16)
        orig_rois = extract_rois_from_frame(orig_f, rois)
        proc_rois = extract_rois_from_frame(proc_f, rois)
 
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
 
        # ---- Visualization ----
        if visualize:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(frame, cmap='gray')
            axs[0].set_title(f'Original Frame {i}')
            axs[1].imshow(proc_uint16, cmap='gray')
            axs[1].set_title(f'Processed Frame {i}')
 
            for ax in axs:
                for (x1, x2, y1, y2) in rois:
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                         edgecolor='red', facecolor='none', linewidth=2)
                    ax.add_patch(rect)
            for a in axs: a.axis('off')
            plt.tight_layout()
            plt.show()
 
        if i % 10 == 0:
            print(f" → Processed frame {i+1}/{orig_frames.shape[0]}")
 
    # ---- Save Metrics ----
    df = pd.DataFrame(metrics)
    df.to_excel(excel_path, index=False)
    print(f"\n✅ ROI Metrics saved to: {excel_path}")
 
    # ---- Save Processed DICOM ----
    ds.PixelData = np.stack(processed_frames).tobytes()
    ds.Rows, ds.Columns = processed_frames[0].shape
    ds.save_as(output_path)
    print(f"✅ Processed DICOM saved: {output_path}")
 
# ------------------- Example Usage -------------------
if __name__ == "__main__":
    input_dcm = r"D:\Rohith\Auto_pixel_shift\APS\output_1\output_0/dsa_results.dcm"
    output_dcm = r"D:\Rohith\Auto_pixel_shift\APS\output_1\output_0\DSA_postprocessed_final.dcm"
    excel_out = r"D:\Rohith\Auto_pixel_shift\APS\output_1\output_0\DSA_postprocessed_metrics.xlsx"
 
    rois = [
        (300, 350, 200, 250),
        (515, 565, 725, 775),
        (640, 690, 440, 490),
        (825, 875, 630, 680)
    ]
 
    process_dicom_with_metrics(input_dcm, output_dcm, excel_out, rois, visualize=True)