"""
Grid-based motion estimation between frames using OpenCV ECC.
Divides frames into grid cells and calculates motion for each cell.
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
from matplotlib.patches import Rectangle

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
    return arr.astype(np.float32)

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
# Grid division functions
# -------------------------
def create_grid_coordinates(height, width, grid_rows, grid_cols):
    """Create grid coordinates for dividing image into cells."""
    row_step = height // grid_rows
    col_step = width // grid_cols
    
    grid_coords = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y_start = row * row_step
            y_end = (row + 1) * row_step if row < grid_rows - 1 else height
            x_start = col * col_step
            x_end = (col + 1) * col_step if col < grid_cols - 1 else width
            
            grid_coords.append({
                'row': row,
                'col': col,
                'grid_id': f'R{row}C{col}',
                'y_start': y_start,
                'y_end': y_end,
                'x_start': x_start,
                'x_end': x_end,
                'height': y_end - y_start,
                'width': x_end - x_start
            })
    
    return grid_coords

def extract_grid_cell(image, grid_info):
    """Extract a single grid cell from the image."""
    return image[grid_info['y_start']:grid_info['y_end'], 
                 grid_info['x_start']:grid_info['x_end']]

# -------------------------
# Warping helpers for grid cells
# -------------------------
def apply_warp(img, warp, motion, dsize):
    """Apply warp to img (float32) given motion type. dsize = (w,h)."""
    if motion == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        return cv2.warpAffine(img, warp, dsize, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

def decompose_warp2d(warp, motion):
    """Decode/approximate the geometric parameters from the warp matrix."""
    if motion == cv2.MOTION_TRANSLATION:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': 0, 'scale_x': 1, 'scale_y': 1, 'shear': 0}
    elif motion == cv2.MOTION_EUCLIDEAN:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        angle_rad = np.arctan2(warp[1,0], warp[0,0])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': float(np.degrees(angle_rad)), 'scale_x': 1, 'scale_y': 1, 'shear': 0}
    elif motion == cv2.MOTION_AFFINE:
        A = warp[:,:2].astype(np.float64)
        tx, ty = float(warp[0,2]), float(warp[1,2])
        sx = np.linalg.norm(A[:,0])
        sy = np.linalg.norm(A[:,1])
        theta = np.arctan2(A[1,0], A[0,0])
        shear = np.dot(A[:,0], A[:,1])/(sx*sy) if sx*sy > 1e-9 else 0.0
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': float(np.degrees(theta)),
                'scale_x': float(sx), 'scale_y': float(sy), 'shear': float(shear)}
    elif motion == cv2.MOTION_HOMOGRAPHY:
        tx, ty = float(warp[0,2]), float(warp[1,2])
        return {'translation_x': tx, 'translation_y': ty, 'rotation_deg': 'N/A', 'scale_x': 'N/A', 'scale_y': 'N/A', 'shear': 'N/A'}
    else:
        return {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 'scale_x': 1, 'scale_y': 1, 'shear': 0}

# -------------------------
# ECC estimation for grid cells
# -------------------------
def estimate_motion_ecc_grid(template_cell, frame_cell, motion=cv2.MOTION_AFFINE,
                            number_of_iterations=200, termination_eps=1e-6,
                            gaussFiltSize=3, min_cell_size=20):
    """
    Estimate ECC for a single grid cell.
    Returns (ecc_value, warp_matrix, success_flag)
    """
    # Check if cell is large enough
    if template_cell.shape[0] < min_cell_size or template_cell.shape[1] < min_cell_size:
        return -1.0, None, False
    
    # Prepare cells
    T = prepare_for_ecc(template_cell)
    I = prepare_for_ecc(frame_cell)
    
    # Check if cells have enough variation
    if np.std(T) < 0.01 or np.std(I) < 0.01:
        return -1.0, None, False
    
    h, w = T.shape[:2]
    
    # Initial warp
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    
    # Run ECC
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    try:
        res = cv2.findTransformECC(T, I, warp, motion, criteria, None, gaussFiltSize)
        if isinstance(res, tuple) or isinstance(res, list):
            ecc_val, warp = res[0], res[1]
        else:
            ecc_val = float(res)
        return float(ecc_val), warp, True
    except cv2.error as e:
        return -1.0, None, False

# -------------------------
# Grid-based motion analysis
# -------------------------
def analyze_frame_grid_motion(template_frame, current_frame, grid_rows, grid_cols, 
                             motion_model, frame_idx):
    """Analyze motion for all grid cells in a frame."""
    height, width = template_frame.shape[:2]
    grid_coords = create_grid_coordinates(height, width, grid_rows, grid_cols)
    
    grid_results = []
    
    for grid_info in grid_coords:
        # Extract grid cells
        template_cell = extract_grid_cell(template_frame, grid_info)
        current_cell = extract_grid_cell(current_frame, grid_info)
        
        if frame_idx == 0:
            # Frame 0 vs itself - perfect alignment
            ecc = 1.0
            if motion_model == cv2.MOTION_HOMOGRAPHY:
                warp = np.eye(3, dtype=np.float32)
            else:
                warp = np.eye(2, 3, dtype=np.float32)
            params = {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 
                     'scale_x': 1, 'scale_y': 1, 'shear': 0}
            success = True
        else:
            # Calculate ECC for this grid cell
            ecc, warp, success = estimate_motion_ecc_grid(
                template_cell, current_cell, motion_model)
            
            if success and warp is not None:
                params = decompose_warp2d(warp, motion_model)
            else:
                params = {'translation_x': 0, 'translation_y': 0, 'rotation_deg': 0, 
                         'scale_x': 1, 'scale_y': 1, 'shear': 0}
        
        # Store results
        result = {
            'frame_idx': frame_idx,
            'grid_id': grid_info['grid_id'],
            'grid_row': grid_info['row'],
            'grid_col': grid_info['col'],
            'x_start': grid_info['x_start'],
            'y_start': grid_info['y_start'],
            'x_end': grid_info['x_end'],
            'y_end': grid_info['y_end'],
            'cell_width': grid_info['width'],
            'cell_height': grid_info['height'],
            'ecc_correlation': ecc,
            'success': success,
            'translation_x': params['translation_x'],
            'translation_y': params['translation_y'],
            'rotation_deg': params['rotation_deg'],
            'scale_x': params['scale_x'],
            'scale_y': params['scale_y'],
            'shear': params['shear'],
            'warp_matrix': warp.tolist() if warp is not None else None
        }
        
        grid_results.append(result)
    
    return grid_results

# -------------------------
# Visualization functions
# -------------------------
def create_grid_visualization(template_frame, current_frame, grid_results, frame_idx, 
                             grid_rows, grid_cols, output_dir):
    """Create comprehensive grid visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to matrices for heatmaps
    ecc_matrix = np.zeros((grid_rows, grid_cols))
    translation_x_matrix = np.zeros((grid_rows, grid_cols))
    translation_y_matrix = np.zeros((grid_rows, grid_cols))
    translation_magnitude_matrix = np.zeros((grid_rows, grid_cols))
    
    for result in grid_results:
        row, col = result['grid_row'], result['grid_col']
        ecc_matrix[row, col] = result['ecc_correlation'] if result['success'] else -1
        translation_x_matrix[row, col] = result['translation_x']
        translation_y_matrix[row, col] = result['translation_y']
        translation_magnitude_matrix[row, col] = np.sqrt(
            result['translation_x']**2 + result['translation_y']**2)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Grid Motion Analysis - Frame {frame_idx}', fontsize=16)
    
    # Original images with grid overlay
    ax = axes[0, 0]
    ax.imshow(template_frame, cmap='gray')
    ax.set_title('Template (Frame 0) with Grid')
    height, width = template_frame.shape
    for i in range(1, grid_rows):
        ax.axhline(y=i*height/grid_rows, color='red', linewidth=1, alpha=0.7)
    for j in range(1, grid_cols):
        ax.axvline(x=j*width/grid_cols, color='red', linewidth=1, alpha=0.7)
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(current_frame, cmap='gray')
    ax.set_title(f'Current Frame ({frame_idx}) with Grid')
    for i in range(1, grid_rows):
        ax.axhline(y=i*height/grid_rows, color='red', linewidth=1, alpha=0.7)
    for j in range(1, grid_cols):
        ax.axvline(x=j*width/grid_cols, color='red', linewidth=1, alpha=0.7)
    ax.axis('off')
    
    # ECC correlation heatmap
    ax = axes[0, 2]
    im1 = ax.imshow(ecc_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('ECC Correlation per Grid Cell')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.colorbar(im1, ax=ax, fraction=0.046)
    
    # Add text annotations for ECC values
    for i in range(grid_rows):
        for j in range(grid_cols):
            text = f'{ecc_matrix[i,j]:.2f}' if ecc_matrix[i,j] > -1 else 'N/A'
            ax.text(j, i, text, ha='center', va='center', fontsize=8)
    
    # Translation X heatmap
    ax = axes[1, 0]
    im2 = ax.imshow(translation_x_matrix, cmap='RdBu_r', 
                    vmin=-np.max(np.abs(translation_x_matrix)), 
                    vmax=np.max(np.abs(translation_x_matrix)))
    ax.set_title('Translation X (pixels)')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.colorbar(im2, ax=ax, fraction=0.046)
    
    # Translation Y heatmap
    ax = axes[1, 1]
    im3 = ax.imshow(translation_y_matrix, cmap='RdBu_r',
                    vmin=-np.max(np.abs(translation_y_matrix)), 
                    vmax=np.max(np.abs(translation_y_matrix)))
    ax.set_title('Translation Y (pixels)')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.colorbar(im3, ax=ax, fraction=0.046)
    
    # Translation magnitude heatmap
    ax = axes[1, 2]
    im4 = ax.imshow(translation_magnitude_matrix, cmap='plasma')
    ax.set_title('Translation Magnitude (pixels)')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.colorbar(im4, ax=ax, fraction=0.046)
    
    # Add text annotations for translation values
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax.text(j, i, f'{translation_magnitude_matrix[i,j]:.1f}', 
                   ha='center', va='center', fontsize=8, color='white')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(output_dir, f'grid_analysis_frame_{frame_idx}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

# -------------------------
# Excel export functions
# -------------------------
def export_grid_results_to_excel(all_results, image_paths, output_file, grid_rows, grid_cols):
    """Export grid-based results to Excel with visualizations."""
    workbook = xlsxwriter.Workbook(output_file)
    
    # Create summary worksheet
    summary_ws = workbook.add_worksheet('Summary')
    
    # Create detailed results worksheet
    detailed_ws = workbook.add_worksheet('Detailed_Grid_Results')
    
    # Headers for detailed results
    detailed_headers = ['Frame_Index', 'Grid_ID', 'Grid_Row', 'Grid_Col', 
                       'X_Start', 'Y_Start', 'X_End', 'Y_End',
                       'Cell_Width', 'Cell_Height', 'ECC_Correlation', 'Success',
                       'Translation_X', 'Translation_Y', 'Rotation_Deg', 
                       'Scale_X', 'Scale_Y', 'Shear']
    
    # Write detailed headers
    for col, header in enumerate(detailed_headers):
        detailed_ws.write(0, col, header)
    
    # Write detailed data
    row_idx = 1
    for frame_results in all_results:
        for result in frame_results:
            detailed_ws.write(row_idx, 0, result['frame_idx'])
            detailed_ws.write(row_idx, 1, result['grid_id'])
            detailed_ws.write(row_idx, 2, result['grid_row'])
            detailed_ws.write(row_idx, 3, result['grid_col'])
            detailed_ws.write(row_idx, 4, result['x_start'])
            detailed_ws.write(row_idx, 5, result['y_start'])
            detailed_ws.write(row_idx, 6, result['x_end'])
            detailed_ws.write(row_idx, 7, result['y_end'])
            detailed_ws.write(row_idx, 8, result['cell_width'])
            detailed_ws.write(row_idx, 9, result['cell_height'])
            detailed_ws.write(row_idx, 10, result['ecc_correlation'])
            detailed_ws.write(row_idx, 11, result['success'])
            detailed_ws.write(row_idx, 12, result['translation_x'])
            detailed_ws.write(row_idx, 13, result['translation_y'])
            detailed_ws.write(row_idx, 14, result['rotation_deg'])
            detailed_ws.write(row_idx, 15, result['scale_x'])
            detailed_ws.write(row_idx, 16, result['scale_y'])
            detailed_ws.write(row_idx, 17, result['shear'])
            row_idx += 1
    
    # Create images worksheet
    images_ws = workbook.add_worksheet('Grid_Visualizations')
    images_ws.write(0, 0, 'Frame_Index')
    images_ws.write(0, 1, 'Grid_Visualization')
    
    # Insert images
    for idx, (frame_idx, image_path) in enumerate(image_paths, start=1):
        images_ws.write(idx, 0, frame_idx)
        if os.path.exists(image_path):
            images_ws.set_row(idx, 300)  # Set row height for large images
            images_ws.insert_image(idx, 1, image_path, {'x_scale': 0.4, 'y_scale': 0.4})
    
    # Summary statistics
    summary_ws.write(0, 0, 'Grid Motion Analysis Summary')
    summary_ws.write(2, 0, f'Grid Configuration: {grid_rows} x {grid_cols}')
    summary_ws.write(3, 0, f'Total Frames Processed: {len(all_results)}')
    summary_ws.write(4, 0, f'Grid Cells per Frame: {grid_rows * grid_cols}')
    
    workbook.close()
    print(f"Excel file saved: {output_file}")

# -------------------------
# Main processing function
# -------------------------
def process_frames_with_grid_analysis(dicom_path, grid_rows=4, grid_cols=4, 
                                     motion_model=cv2.MOTION_AFFINE,
                                     output_dir="grid_motion_analysis"):
    """Process all frames with grid-based motion analysis."""
    
    # Load frames
    frames = read_dicom_frames(dicom_path)
    print(f"Loaded frames: {frames.shape}")
    
    if frames.shape[0] < 2:
        print("Need at least 2 frames for motion estimation")
        return
    
    # Use frame 0 as template
    template_frame = prepare_for_ecc(frames[0])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "grid_visualizations")
    
    # Results storage
    all_results = []
    image_paths = []
    
    print(f"Grid configuration: {grid_rows} x {grid_cols} = {grid_rows * grid_cols} cells per frame")
    print("="*60)
    
    # Process each frame
    for frame_idx in range(frames.shape[0]):
        print(f"Processing frame {frame_idx}...")
        
        current_frame = prepare_for_ecc(frames[frame_idx])
        
        # Analyze grid motion for this frame
        grid_results = analyze_frame_grid_motion(
            template_frame, current_frame, grid_rows, grid_cols, 
            motion_model, frame_idx)
        
        all_results.append(grid_results)
        
        # Create visualization
        try:
            viz_path = create_grid_visualization(
                template_frame, current_frame, grid_results, frame_idx,
                grid_rows, grid_cols, images_dir)
            image_paths.append((frame_idx, viz_path))
        except Exception as e:
            print(f"  Warning: Could not create visualization for frame {frame_idx}: {e}")
            image_paths.append((frame_idx, ""))
        
        # Print summary for this frame
        successful_cells = sum(1 for result in grid_results if result['success'])
        avg_ecc = np.mean([r['ecc_correlation'] for r in grid_results if r['success']])
        avg_translation = np.mean([np.sqrt(r['translation_x']**2 + r['translation_y']**2) 
                                  for r in grid_results if r['success']])
        
        print(f"  Successful cells: {successful_cells}/{len(grid_results)}")
        if successful_cells > 0:
            print(f"  Average ECC: {avg_ecc:.3f}")
            print(f"  Average translation magnitude: {avg_translation:.2f} pixels")
    
    # Export to CSV and Excel
    print("\nExporting results...")
    
    # Flatten results for CSV
    flat_results = []
    for frame_results in all_results:
        flat_results.extend(frame_results)
    
    df = pd.DataFrame(flat_results)
    
    # Save CSV
    csv_file = os.path.join(output_dir, f"grid_motion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")
    
    # Save Excel with visualizations
    excel_file = os.path.join(output_dir, f"grid_motion_analysis_with_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    try:
        export_grid_results_to_excel(all_results, image_paths, excel_file, grid_rows, grid_cols)
    except Exception as e:
        print(f"Error creating Excel file: {e}")
    
    return all_results, image_paths

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Configuration
    dicom_path = "D:/Rohith/RAW_AUTOPIXEL/GI_RAW/1.2.826.0.1.3680043.2.1330.2641710.2312141352140001.5.454_Raw_anon.dcm"
    grid_rows = 1344       # Number of rows in grid
    grid_cols = 1344       # Number of columns in grid
    motion_model = cv2.MOTION_AFFINE  # Motion model
    output_directory = "grid_motion_analysis_results"
    
    print("="*60)
    print("GRID-BASED MOTION ESTIMATION ANALYSIS")
    print("="*60)
    print(f"DICOM file: {dicom_path}")
    print(f"Grid size: {grid_rows} x {grid_cols}")
    print(f"Motion model: {motion_model}")
    print(f"Output directory: {output_directory}")
    print("="*60)
    
    # Process all frames with grid analysis
    try:
        all_results, image_paths = process_frames_with_grid_analysis(
            dicom_path, grid_rows, grid_cols, motion_model, output_directory) # type: ignore
        
        print("\n" + "="*60)
        print("GRID ANALYSIS COMPLETE!")
        print("="*60)
        print("Results summary:")
        print(f"- Total frames processed: {len(all_results)}")
        print(f"- Grid cells per frame: {grid_rows * grid_cols}")
        print(f"- Total grid cells analyzed: {len(all_results) * grid_rows * grid_cols}")
        print(f"- Output directory: {output_directory}")
        print("- Files created:")
        print("  * CSV file with detailed grid results")
        print("  * Excel file with embedded grid visualizations")
        print("  * Individual visualization images")
        print("="*60)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()