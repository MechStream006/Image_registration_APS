"""
Enhanced motion estimation between frames with Excel export and embedded images.
Requirements:
    pip install opencv-python pydicom matplotlib numpy pandas openpyxl xlsxwriter Pillow
"""

import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import base64
from PIL import Image
import xlsxwriter
from datetime import datetime

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
    # convert to float32
    return arr.astype(np.float32)

def read_image_as_gray(path):
    """Read a single image (png/jpg/...) and return float32 grayscale."""
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
# Warping helpers
# -------------------------
def apply_warp(img, warp, motion, dsize):
    """Apply warp to img (float32) given motion type. dsize = (w,h)."""
    if motion == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        return cv2.warpAffine(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# -------------------------
# Decompose warp matrix
# -------------------------
def decompose_warp2d(warp, motion):
    """
    Decode/approximate the geometric parameters from the warp matrix.
    Returns a dict with keys depending on motion type.
    """
    if motion == cv2.MOTION_TRANSLATION:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': 0, 'scale_x': 1, 'scale_y': 1, 'shear': 0}
    elif motion == cv2.MOTION_EUCLIDEAN:
        # warp is [ cos -sin tx; sin cos ty ]
        tx, ty = float(warp[0,2]), float(warp[1,2])
        angle_rad = np.arctan2(warp[1,0], warp[0,0])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': float(np.degrees(angle_rad)), 'scale_x': 1, 'scale_y': 1, 'shear': 0}
    elif motion == cv2.MOTION_AFFINE:
        A = warp[:,:2].astype(np.float64)  # 2x2
        tx, ty = float(warp[0,2]), float(warp[1,2])
        # scales (approx)
        sx = np.linalg.norm(A[:,0])
        sy = np.linalg.norm(A[:,1])
        # rotation approx from first column
        theta = np.arctan2(A[1,0], A[0,0])
        # shear (cosine between columns)
        shear = np.dot(A[:,0], A[:,1])/(sx*sy) if sx*sy > 1e-9 else 0.0
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': float(np.degrees(theta)),
                'scale_x': float(sx), 'scale_y': float(sy), 'shear': float(shear)}
    elif motion == cv2.MOTION_HOMOGRAPHY:
        # For homography, just return translation components and mark others as N/A
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': 'N/A', 'scale_x': 'N/A', 'scale_y': 'N/A', 'shear': 'N/A'}
    else:
        return {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 'scale_x': 1, 'scale_y': 1, 'shear': 0}

# -------------------------
# ECC estimation (optionally multi-scale)
# -------------------------
def estimate_motion_ecc(template, frame, motion=cv2.MOTION_AFFINE,
                        number_of_iterations=500, termination_eps=1e-6,
                        gaussFiltSize=5, pyr_levels=0, init_warp=None, mask=None):
    """
    Estimate warp that maps `frame` -> `template` using ECC.
    If pyr_levels > 0, run from coarse to fine (pyramid).
    Returns (ecc_value, warp_matrix)
    """
    # prepare images
    T = prepare_for_ecc(template)
    I = prepare_for_ecc(frame)
    h, w = T.shape[:2]

    # initial warp
    if motion == cv2.MOTION_HOMOGRAPHY:
        if init_warp is None:
            warp = np.eye(3, dtype=np.float32)
        else:
            warp = init_warp.astype(np.float32)
    else:
        if init_warp is None:
            warp = np.eye(2, 3, dtype=np.float32)
        else:
            warp = init_warp.astype(np.float32)

    levels = list(range(pyr_levels, -1, -1))  
    current_warp = warp.copy()

    for level in levels:
        scale = 1.0 / (2 ** level)
        Tw = cv2.resize(T, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        Iw = cv2.resize(I, (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_AREA)
        mask_w = None
        if mask is not None:
            mask_w = cv2.resize(mask.astype(np.uint8), (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_NEAREST)

        if level != pyr_levels: 
            pass  

        # prepare warp for this level:
        if motion == cv2.MOTION_HOMOGRAPHY:
            if current_warp.shape != (3,3):
                tmp = np.eye(3, dtype=np.float32)
                tmp[:2,:] = current_warp
                current_warp = tmp
        else:
            if current_warp.shape != (2,3):
                tmp = np.eye(2,3, dtype=np.float32)
                tmp[:2,:] = current_warp[:2,:]
                current_warp = tmp

        if level != levels[0]:  
            if motion == cv2.MOTION_HOMOGRAPHY:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0
            else:
                current_warp[0,2] *= 2.0
                current_warp[1,2] *= 2.0

        # run ECC on Tw (template) and Iw (input)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        try:
            res = cv2.findTransformECC(Tw, Iw, current_warp, motion, criteria, mask_w, gaussFiltSize)
            if isinstance(res, tuple) or isinstance(res, list):
                ecc_val, current_warp = res[0], res[1]
            else:
                ecc_val = float(res)
        except cv2.error as e:
            print(f"Warning: ECC failed for level {level}: {e}")
            ecc_val = -1.0  # indicate failure
            break

    return float(ecc_val), current_warp

# -------------------------
# Image generation for Excel
# -------------------------
def create_alignment_images(template, frame, warp, motion, frame_idx, output_dir):
    """Create alignment visualization images and return file paths."""
    h, w = template.shape[:2]
    aligned = apply_warp(frame, warp, motion, (w, h))
    diff = template - aligned
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save aligned image
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template (Frame 0)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(frame, cmap='gray')
    plt.title(f'Input (Frame {frame_idx})')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(aligned, cmap='gray')
    plt.title('Aligned')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(diff, cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.title('Difference (Template - Aligned)')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    aligned_path = os.path.join(output_dir, f'alignment_frame_{frame_idx}.png')
    plt.savefig(aligned_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create difference image separately
    plt.figure(figsize=(6, 4))
    plt.imshow(diff, cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.title(f'Difference: Template - Frame {frame_idx}')
    plt.colorbar()
    plt.axis('off')
    diff_path = os.path.join(output_dir, f'difference_frame_{frame_idx}.png')
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return aligned_path, diff_path

# -------------------------
# Excel export with images
# -------------------------
def export_to_excel_with_images(results_df, image_paths, output_file):
    """Export results to Excel with embedded images."""
    # Create Excel writer
    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet('Motion_Analysis')
    
    # Write headers
    headers = ['Frame_Index', 'ECC_Correlation', 'Warp_Matrix', 'Translation_X', 'Translation_Y', 
               'Rotation_Deg', 'Scale_X', 'Scale_Y', 'Shear', 'Alignment_Image', 'Difference_Image']
    
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    
    # Set column widths
    worksheet.set_column('A:I', 15)
    worksheet.set_column('J:K', 30)  # Wider for image columns
    
    # Write data and insert images
    for idx, (_, row) in enumerate(results_df.iterrows(), start=1):
        worksheet.write(idx, 0, row['Frame_Index'])
        worksheet.write(idx, 1, row['ECC_Correlation'])
        worksheet.write(idx, 2, str(row['Warp_Matrix']))
        worksheet.write(idx, 3, row['Translation_X'])
        worksheet.write(idx, 4, row['Translation_Y'])
        worksheet.write(idx, 5, row['Rotation_Deg'])
        worksheet.write(idx, 6, row['Scale_X'])
        worksheet.write(idx, 7, row['Scale_Y'])
        worksheet.write(idx, 8, row['Shear'])
        
        # Insert images if they exist
        if idx-1 < len(image_paths):
            aligned_path, diff_path = image_paths[idx-1]
            if os.path.exists(aligned_path):
                worksheet.set_row(idx, 180)  # Set row height for images
                worksheet.insert_image(idx, 9, aligned_path, {'x_scale': 0.3, 'y_scale': 0.3})
            if os.path.exists(diff_path):
                worksheet.insert_image(idx, 10, diff_path, {'x_scale': 0.3, 'y_scale': 0.3})
    
    workbook.close()
    print(f"Excel file saved: {output_file}")

# -------------------------
# Main processing function
# -------------------------
def process_all_frames_with_excel_export(dicom_path, motion_model=cv2.MOTION_AFFINE, 
                                        pyr_levels=1, output_dir="motion_analysis_output"):
    """Process all frames against frame 0 and export to Excel with images."""
    
    # Load frames
    frames = read_dicom_frames(dicom_path)
    print(f"Loaded frames: {frames.shape}")
    
    if frames.shape[0] < 2:
        print("Need at least 2 frames for motion estimation")
        return
    
    # Use frame 0 as template (mask frame)
    template = frames[0]
    template_prepared = prepare_for_ecc(template)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    
    # Results storage
    results = []
    image_paths = []
    
    # Process each frame against template
    for frame_idx in range(frames.shape[0]):
        print(f"Processing frame {frame_idx}...")
        
        current_frame = frames[frame_idx]
        current_prepared = prepare_for_ecc(current_frame)
        
        if frame_idx == 0:
            # Frame 0 vs itself - perfect alignment
            ecc = 1.0
            if motion_model == cv2.MOTION_HOMOGRAPHY:
                warp = np.eye(3, dtype=np.float32)
            else:
                warp = np.eye(2, 3, dtype=np.float32)
            params = {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 
                     'scale_x': 1, 'scale_y': 1, 'shear': 0}
        else:
            try:
                ecc, warp = estimate_motion_ecc(template_prepared, current_prepared, 
                                              motion=motion_model, pyr_levels=pyr_levels,
                                              number_of_iterations=200, termination_eps=1e-6, 
                                              gaussFiltSize=5)
                params = decompose_warp2d(warp, motion_model)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                ecc = -1.0
                if motion_model == cv2.MOTION_HOMOGRAPHY:
                    warp = np.eye(3, dtype=np.float32)
                else:
                    warp = np.eye(2, 3, dtype=np.float32)
                params = {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 
                         'scale_x': 1, 'scale_y': 1, 'shear': 0}
        
        # Create visualization images
        try:
            aligned_path, diff_path = create_alignment_images(
                template_prepared, current_prepared, warp, motion_model, frame_idx, images_dir)
            image_paths.append((aligned_path, diff_path))
        except Exception as e:
            print(f"Error creating images for frame {frame_idx}: {e}")
            image_paths.append(("", ""))
        
        # Store results
        result = {
            'Frame_Index': frame_idx,
            'ECC_Correlation': ecc,
            'Warp_Matrix': warp.tolist(),
            'Translation_X': params.get('translation_x', 0),
            'Translation_Y': params.get('translation_y', 0),
            'Rotation_Deg': params.get('rotation_deg', 0),
            'Scale_X': params.get('scale_x', 1),
            'Scale_Y': params.get('scale_y', 1),
            'Shear': params.get('shear', 0)
        }
        results.append(result)
        
        # Print results for current frame
        print(f"  Frame {frame_idx} - ECC: {ecc:.4f}")
        print(f"  Translation: ({params.get('translation_x', 0):.2f}, {params.get('translation_y', 0):.2f})")
        print(f"  Rotation: {params.get('rotation_deg', 0):.2f}°")
        if isinstance(params.get('scale_x', 1), (int, float)):
            print(f"  Scale: ({params.get('scale_x', 1):.3f}, {params.get('scale_y', 1):.3f})")
            print(f"  Shear: {params.get('shear', 0):.3f}")
    
    # Create DataFrame and export to Excel
    results_df = pd.DataFrame(results)
    
    # Export to CSV first (simpler format)
    csv_file = os.path.join(output_dir, f"motion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")
    
    # Export to Excel with images
    excel_file = os.path.join(output_dir, f"motion_analysis_with_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    try:
        export_to_excel_with_images(results_df, image_paths, excel_file)
    except Exception as e:
        print(f"Error creating Excel file with images: {e}")
        print("CSV file has been created successfully.")
    
    return results_df, image_paths

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Configuration
    dicom_path = "D:/Rohith/RAW_AUTOPIXEL/GI_RAW/1.2.826.0.1.3680043.2.1330.2641710.2312141414290002.5.466_Raw_anon.dcm"
    motion_model = cv2.MOTION_AFFINE  # MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY
    pyr_levels = 1  # 0 = single scale; 1-2 helpful for larger motions
    output_directory = "motion_analysis_results"
    
    print("="*60)
    print("MOTION ESTIMATION ANALYSIS")
    print("="*60)
    print(f"DICOM file: {dicom_path}")
    print(f"Motion model: {motion_model}")
    print(f"Pyramid levels: {pyr_levels}")
    print(f"Output directory: {output_directory}")
    print("="*60)
    
    # Process all frames
    try:
        results_df, image_paths = process_all_frames_with_excel_export(
            dicom_path, motion_model, pyr_levels, output_directory)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Results summary:")
        print(f"- Total frames processed: {len(results_df)}")
        print(f"- Output directory: {output_directory}")
        print(f"- Images saved in: {os.path.join(output_directory, 'images')}")
        print("- Files created:")
        print("  * CSV file with numerical results")
        print("  * Excel file with embedded images")
        print("="*60)
     
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()