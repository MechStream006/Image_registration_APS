import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIGURATION
# ============================================================
dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
output_dir = Path("D:/Rohith/Auto_pixel_shift/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Processing parameters
MASK_FRAME_IDX = 0  # First frame as mask
START_FRAME = 1     # Start processing from frame 1
END_FRAME = None    # None = process all frames, or set specific number

# Feature detection parameters
ORB_FEATURES = 2000
RATIO_TEST_THRESHOLD = 0.75
MIN_MATCHES = 4
RANSAC_THRESHOLD = 2.0

# Temporal filtering parameters
TEMPORAL_SMOOTHING = True
SMOOTHING_SIGMA = 1.5  # Higher = more smoothing

# Visualization parameters
SAVE_VISUALIZATIONS = True
SHOW_PLOTS = True
VIS_SAMPLE_INTERVAL = 5  # Visualize every Nth frame

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_frame(frame):
    """Enhance frame for better feature detection"""
    if frame.max() > 255:
        frame_norm = cv2.convertScaleAbs(frame, alpha=(255.0 / frame.max()))
    else:
        frame_norm = frame.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(frame_norm)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def estimate_motion(mask_frame, live_frame, orb, log_list):
    """
    Estimate motion between mask and live frame
    Returns: transformation matrix, number of inliers, inlier ratio
    """
    # Detect features
    kp_mask, des_mask = orb.detectAndCompute(mask_frame, None)
    kp_live, des_live = orb.detectAndCompute(live_frame, None)
    
    log_entry = {
        'mask_keypoints': len(kp_mask) if kp_mask else 0,
        'live_keypoints': len(kp_live) if kp_live else 0,
    }
    
    # Check if enough features
    if des_mask is None or des_live is None or len(des_mask) < 4 or len(des_live) < 4:
        log_entry['status'] = 'insufficient_features'
        log_entry['transformation'] = 'identity'
        log_list.append(log_entry)
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0, 0.0
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_mask, des_live, k=2)
    
    # Ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)
    
    log_entry['total_matches'] = len(matches)
    log_entry['good_matches'] = len(good_matches)
    
    # Check minimum matches
    if len(good_matches) < MIN_MATCHES:
        log_entry['status'] = 'insufficient_matches'
        log_entry['transformation'] = 'identity'
        log_list.append(log_entry)
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0, 0.0
    
    # Extract points
    pts_mask = np.float32([kp_mask[m.queryIdx].pt for m in good_matches])
    pts_live = np.float32([kp_live[m.trainIdx].pt for m in good_matches])
    
    # Estimate transformation
    M, inliers = cv2.estimateAffine2D(
        pts_mask, pts_live,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99
    )
    
    if M is None:
        log_entry['status'] = 'estimation_failed'
        log_entry['transformation'] = 'identity'
        log_list.append(log_entry)
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0, 0.0
    
    inlier_count = np.sum(inliers) if inliers is not None else 0
    inlier_ratio = inlier_count / len(good_matches) if len(good_matches) > 0 else 0.0
    
    log_entry['status'] = 'success'
    log_entry['inliers'] = int(inlier_count)
    log_entry['inlier_ratio'] = float(inlier_ratio)
    log_entry['dx'] = float(M[0, 2])
    log_entry['dy'] = float(M[1, 2])
    log_entry['rotation_deg'] = float(np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi)
    log_entry['scale_x'] = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
    log_entry['scale_y'] = float(np.sqrt(M[0, 1]**2 + M[1, 1]**2))
    
    log_list.append(log_entry)
    
    return M, inlier_count, inlier_ratio

def smooth_motion_parameters(transformations):
    """
    Apply temporal smoothing to motion parameters
    """
    if len(transformations) < 3:
        return transformations
    
    # Extract parameters
    dx_list = [M[0, 2] for M in transformations]
    dy_list = [M[1, 2] for M in transformations]
    
    # Smooth translations
    dx_smooth = gaussian_filter1d(dx_list, sigma=SMOOTHING_SIGMA)
    dy_smooth = gaussian_filter1d(dy_list, sigma=SMOOTHING_SIGMA)
    
    # Reconstruct transformation matrices
    smoothed = []
    for i, M in enumerate(transformations):
        M_smooth = M.copy()
        M_smooth[0, 2] = dx_smooth[i]
        M_smooth[1, 2] = dy_smooth[i]
        smoothed.append(M_smooth)
    
    return smoothed

def save_visualization(frame_idx, mask_frame, live_frame, aligned_mask, subtraction, M, log_entry, output_dir):
    """Save visualization for specific frame"""
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Original frames
    plt.subplot(3, 4, 1)
    plt.title(f"Mask Frame {MASK_FRAME_IDX}")
    plt.imshow(mask_frame, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.title(f"Live Frame {frame_idx}")
    plt.imshow(live_frame, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.title("Aligned Mask")
    plt.imshow(aligned_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.title("DSA Subtraction")
    plt.imshow(subtraction, cmap='gray')
    plt.axis('off')
    
    # Row 2: Difference images
    plt.subplot(3, 4, 5)
    diff_before = cv2.absdiff(mask_frame, live_frame)
    plt.title("Diff Before Alignment")
    plt.imshow(diff_before, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    diff_after = cv2.absdiff(aligned_mask, live_frame)
    plt.title("Diff After Alignment")
    plt.imshow(diff_after, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.title("Overlay Before")
    overlay_before = cv2.addWeighted(mask_frame, 0.5, live_frame, 0.5, 0)
    plt.imshow(overlay_before, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.title("Overlay After")
    overlay_after = cv2.addWeighted(aligned_mask, 0.5, live_frame, 0.5, 0)
    plt.imshow(overlay_after, cmap='gray')
    plt.axis('off')
    
    # Row 3: Metrics and info
    plt.subplot(3, 4, 9)
    plt.axis('off')
    info_text = f"""Frame: {frame_idx}
Status: {log_entry.get('status', 'N/A')}
Keypoints: {log_entry.get('live_keypoints', 0)}
Good Matches: {log_entry.get('good_matches', 0)}
Inliers: {log_entry.get('inliers', 0)}
Inlier Ratio: {log_entry.get('inlier_ratio', 0):.2%}"""
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.subplot(3, 4, 10)
    plt.axis('off')
    motion_text = f"""Motion Parameters:
dx: {log_entry.get('dx', 0):.2f} px
dy: {log_entry.get('dy', 0):.2f} px
Rotation: {log_entry.get('rotation_deg', 0):.2f}°
Scale X: {log_entry.get('scale_x', 1):.3f}
Scale Y: {log_entry.get('scale_y', 1):.3f}"""
    plt.text(0.1, 0.5, motion_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.subplot(3, 4, 11)
    plt.title("Subtraction Enhanced")
    sub_enhanced = cv2.equalizeHist(subtraction)
    plt.imshow(sub_enhanced, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.title("Vessels (Contrast)")
    # Highlight vessels by thresholding subtraction
    _, vessels = cv2.threshold(subtraction, 30, 255, cv2.THRESH_BINARY)
    plt.imshow(vessels, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"frame_{frame_idx:04d}_analysis.png", dpi=150, bbox_inches='tight')
    
    if SHOW_PLOTS and frame_idx % VIS_SAMPLE_INTERVAL == 0:
        plt.show()
    else:
        plt.close()

# ============================================================
# MAIN PROCESSING
# ============================================================

print("="*70)
print("DSA AUTOMATIC PIXEL SHIFT - COMPLETE PROCESSING PIPELINE")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {output_dir}")
print()

# Read DICOM
print("Loading DICOM file...")
ds = pydicom.dcmread(dicom_path)
frames = ds.pixel_array.astype(np.float32)

total_frames = frames.shape[0]
h, w = frames.shape[1], frames.shape[2]

if END_FRAME is None:
    END_FRAME = total_frames

print(f"Total frames: {total_frames}")
print(f"Frame size: {h} × {w}")
print(f"Processing frames: {START_FRAME} to {END_FRAME}")
print()

# Initialize ORB detector
orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)

# Preprocess mask frame
print("Preprocessing mask frame...")
mask_frame = preprocess_frame(frames[MASK_FRAME_IDX])

# Initialize storage
aligned_frames = []
subtracted_frames = []
transformations = []
processing_log = []

# Process each frame
print("\nProcessing frames...")
print("-"*70)

for frame_idx in range(START_FRAME, END_FRAME):
    print(f"Processing frame {frame_idx}/{END_FRAME-1}...", end=" ")
    
    # Preprocess live frame
    live_frame = preprocess_frame(frames[frame_idx])
    
    # Estimate motion
    log_entry = {'frame': frame_idx}
    M, inlier_count, inlier_ratio = estimate_motion(mask_frame, live_frame, orb, [log_entry])
    log_entry = log_entry if not processing_log else processing_log[-1]
    
    transformations.append(M)
    processing_log.append(log_entry)
    
    # Status
    status = log_entry.get('status', 'unknown')
    if status == 'success':
        print(f"✓ [{inlier_count} inliers, {inlier_ratio:.1%}]")
    else:
        print(f"⚠ [{status}]")

# Apply temporal smoothing
if TEMPORAL_SMOOTHING:
    print("\nApplying temporal smoothing to motion parameters...")
    transformations_raw = transformations.copy()
    transformations = smooth_motion_parameters(transformations)
    
    # Update log with smoothed values
    for i, M in enumerate(transformations):
        processing_log[i]['dx_smoothed'] = float(M[0, 2])
        processing_log[i]['dy_smoothed'] = float(M[1, 2])

# Apply transformations and perform subtraction
print("\nApplying transformations and performing DSA subtraction...")
for i, (frame_idx, M) in enumerate(zip(range(START_FRAME, END_FRAME), transformations)):
    live_frame = preprocess_frame(frames[frame_idx])
    
    # Align mask
    aligned_mask = cv2.warpAffine(mask_frame, M, (w, h), flags=cv2.INTER_LINEAR)
    aligned_frames.append(aligned_mask)
    
    # Perform subtraction
    subtraction = cv2.subtract(live_frame, aligned_mask)
    subtracted_frames.append(subtraction)
    
    # Calculate quality metrics
    mse_before = np.mean((mask_frame.astype(float) - live_frame.astype(float))**2)
    mse_after = np.mean((aligned_mask.astype(float) - live_frame.astype(float))**2)
    improvement = ((mse_before - mse_after) / mse_before) * 100 if mse_before > 0 else 0
    
    processing_log[i]['mse_before'] = float(mse_before)
    processing_log[i]['mse_after'] = float(mse_after)
    processing_log[i]['improvement_pct'] = float(improvement)
    
    # Save visualization for sample frames
    if SAVE_VISUALIZATIONS and (frame_idx % VIS_SAMPLE_INTERVAL == 0 or frame_idx == START_FRAME):
        save_visualization(frame_idx, mask_frame, live_frame, aligned_mask, 
                         subtraction, M, processing_log[i], output_dir)

# Convert to numpy arrays
aligned_frames = np.array(aligned_frames, dtype=np.uint8)
subtracted_frames = np.array(subtracted_frames, dtype=np.uint8)

print(f"\n✓ Processed {len(aligned_frames)} frames")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving results...")

# 1. Save aligned frames as DICOM
print("  - Saving aligned cine loop...")
ds_aligned = ds.copy()
ds_aligned.PixelData = aligned_frames.tobytes()
ds_aligned.NumberOfFrames = len(aligned_frames)
ds_aligned.SeriesDescription = "Aligned Mask Frames"
aligned_dcm_path = output_dir / "aligned_mask_cine.dcm"
ds_aligned.save_as(aligned_dcm_path)
print(f"    Saved: {aligned_dcm_path}")

# 2. Save DSA subtracted frames as DICOM
print("  - Saving DSA subtraction cine loop...")
ds_subtracted = ds.copy()
ds_subtracted.PixelData = subtracted_frames.tobytes()
ds_subtracted.NumberOfFrames = len(subtracted_frames)
ds_subtracted.SeriesDescription = "DSA Subtraction"
subtracted_dcm_path = output_dir / "dsa_subtraction_cine.dcm"
ds_subtracted.save_as(subtracted_dcm_path)
print(f"    Saved: {subtracted_dcm_path}")

# 3. Save transformation matrices
print("  - Saving transformation matrices...")
transformations_list = [M.tolist() for M in transformations]
with open(output_dir / "transformation_matrices.json", 'w') as f:
    json.dump(transformations_list, f, indent=2)

# 4. Save processing log
print("  - Saving processing log...")
with open(output_dir / "processing_log.json", 'w') as f:
    json.dump(processing_log, f, indent=2)

# 5. Save summary statistics
print("  - Saving summary statistics...")
successful = sum(1 for log in processing_log if log.get('status') == 'success')
avg_inlier_ratio = np.mean([log.get('inlier_ratio', 0) for log in processing_log if log.get('status') == 'success'])
avg_improvement = np.mean([log.get('improvement_pct', 0) for log in processing_log])

summary = {
    'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_file': str(dicom_path),
    'total_frames': total_frames,
    'processed_frames': len(processing_log),
    'successful_registrations': successful,
    'failed_registrations': len(processing_log) - successful,
    'average_inlier_ratio': float(avg_inlier_ratio),
    'average_mse_improvement_pct': float(avg_improvement),
    'temporal_smoothing': TEMPORAL_SMOOTHING,
    'smoothing_sigma': SMOOTHING_SIGMA if TEMPORAL_SMOOTHING else None,
    'parameters': {
        'mask_frame': MASK_FRAME_IDX,
        'orb_features': ORB_FEATURES,
        'ratio_test_threshold': RATIO_TEST_THRESHOLD,
        'ransac_threshold': RANSAC_THRESHOLD
    }
}

with open(output_dir / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# ============================================================
# GENERATE SUMMARY PLOTS
# ============================================================

print("\nGenerating summary plots...")

# Plot 1: Motion parameters over time
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

frames_processed = list(range(START_FRAME, END_FRAME))
dx_values = [log.get('dx', 0) for log in processing_log]
dy_values = [log.get('dy', 0) for log in processing_log]
inlier_ratios = [log.get('inlier_ratio', 0) for log in processing_log]

axes[0].plot(frames_processed, dx_values, 'b-', linewidth=2, label='dx')
if TEMPORAL_SMOOTHING:
    dx_smooth = [log.get('dx_smoothed', log.get('dx', 0)) for log in processing_log]
    axes[0].plot(frames_processed, dx_smooth, 'r--', linewidth=2, label='dx (smoothed)')
axes[0].set_ylabel('X Translation (pixels)', fontsize=12)
axes[0].set_title('Motion Parameters Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(frames_processed, dy_values, 'g-', linewidth=2, label='dy')
if TEMPORAL_SMOOTHING:
    dy_smooth = [log.get('dy_smoothed', log.get('dy', 0)) for log in processing_log]
    axes[1].plot(frames_processed, dy_smooth, 'r--', linewidth=2, label='dy (smoothed)')
axes[1].set_ylabel('Y Translation (pixels)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(frames_processed, inlier_ratios, 'm-', linewidth=2)
axes[2].axhline(y=0.3, color='r', linestyle='--', label='Warning threshold')
axes[2].set_ylabel('Inlier Ratio', fontsize=12)
axes[2].set_xlabel('Frame Number', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig(output_dir / "motion_parameters_timeline.png", dpi=150, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# Plot 2: Registration quality
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MSE improvement
improvements = [log.get('improvement_pct', 0) for log in processing_log]
axes[0, 0].bar(frames_processed, improvements, color='steelblue')
axes[0, 0].set_ylabel('MSE Improvement (%)', fontsize=12)
axes[0, 0].set_xlabel('Frame Number', fontsize=12)
axes[0, 0].set_title('Registration Quality (MSE Improvement)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Inlier count
inlier_counts = [log.get('inliers', 0) for log in processing_log]
axes[0, 1].bar(frames_processed, inlier_counts, color='seagreen')
axes[0, 1].set_ylabel('Inlier Count', fontsize=12)
axes[0, 1].set_xlabel('Frame Number', fontsize=12)
axes[0, 1].set_title('Feature Matching Inliers', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Status distribution
statuses = [log.get('status', 'unknown') for log in processing_log]
status_counts = {}
for status in statuses:
    status_counts[status] = status_counts.get(status, 0) + 1

axes[1, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Registration Status Distribution', fontsize=12, fontweight='bold')

# Histogram of translations
all_translations = np.sqrt(np.array(dx_values)**2 + np.array(dy_values)**2)
axes[1, 1].hist(all_translations, bins=30, color='coral', edgecolor='black')
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_xlabel('Translation Magnitude (pixels)', fontsize=12)
axes[1, 1].set_title('Distribution of Motion Magnitude', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "registration_quality_summary.png", dpi=150, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("SUMMARY:")
print(f"  Total frames processed: {len(processing_log)}")
print(f"  Successful registrations: {successful} ({successful/len(processing_log)*100:.1f}%)")
print(f"  Average inlier ratio: {avg_inlier_ratio:.2%}")
print(f"  Average MSE improvement: {avg_improvement:.1f}%")
print()
print("OUTPUT FILES:")
print(f"  ✓ Aligned cine loop: {aligned_dcm_path.name}")
print(f"  ✓ DSA subtraction: {subtracted_dcm_path.name}")
print(f"  ✓ Transformation matrices: transformation_matrices.json")
print(f"  ✓ Processing log: processing_log.json")
print(f"  ✓ Summary: summary.json")
print(f"  ✓ Motion timeline plot: motion_parameters_timeline.png")
print(f"  ✓ Quality summary plot: registration_quality_summary.png")
if SAVE_VISUALIZATIONS:
    print(f"  ✓ Frame visualizations: frame_XXXX_analysis.png")
print()
print(f"All files saved in: {output_dir}")
print("="*70)