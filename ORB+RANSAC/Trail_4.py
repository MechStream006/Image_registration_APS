import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === 1. Read DICOM cine loop ===
dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
ds = pydicom.dcmread(dicom_path)

# DICOM cine loops store frames in a 3D pixel array (frames × rows × cols)
frames = ds.pixel_array.astype(np.float32)

print(f"Total frames: {frames.shape[0]}")
print(f"Frame size: {frames.shape[1]} × {frames.shape[2]}")

# === 2. Preprocessing function ===
def preprocess_frame(frame):
    """
    Enhance frame for better feature detection
    """
    # Normalize to 8-bit range
    if frame.max() > 255:
        frame_norm = cv2.convertScaleAbs(frame, alpha=(255.0 / frame.max()))
    else:
        frame_norm = frame.astype(np.uint8)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(frame_norm)
    
    # Optional: Slight blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

# === 3. Select mask and live frames ===
mask_frame_idx = 0
live_frame_idx = 10

mask_frame = preprocess_frame(frames[mask_frame_idx])
live_frame = preprocess_frame(frames[live_frame_idx])

print(f"\nProcessing: Mask frame {mask_frame_idx} vs Live frame {live_frame_idx}")

# === 4. Apply ORB Feature Detector ===
orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)

# Detect keypoints and descriptors
kp_mask, des_mask = orb.detectAndCompute(mask_frame, None)
kp_live, des_live = orb.detectAndCompute(live_frame, None)

print(f"Mask frame: {len(kp_mask)} keypoints")
print(f"Live frame: {len(kp_live)} keypoints")

# === 5. Check if enough features detected ===
if des_mask is None or des_live is None or len(des_mask) < 4 or len(des_live) < 4:
    print("❌ Error: Not enough features detected!")
    print("Try adjusting ORB parameters or check image quality")
    exit()

# === 6. Feature Matching with Ratio Test ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des_mask, des_live, k=2)

# Apply Lowe's ratio test
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:  # Ensure we have 2 matches
        m, n = match_pair
        if m.distance < 0.90 * n.distance:  # Standard ratio threshold
            good_matches.append(m)

print(f"\nTotal matches: {len(matches)}")
print(f"Good matches after ratio test: {len(good_matches)}")

# === 7. Check minimum matches ===
if len(good_matches) < 4:
    print("❌ Error: Not enough good matches for transformation estimation!")
    print("Need at least 4 matches")
    exit()

# === 8. Extract matched keypoint coordinates ===
pts_mask = np.float32([kp_mask[m.queryIdx].pt for m in good_matches])
pts_live = np.float32([kp_live[m.trainIdx].pt for m in good_matches])

# === 9. Estimate transformation using RANSAC ===
# Option A: Full affine (translation + rotation + scale)
M_affine, inliers = cv2.estimateAffine2D(
    pts_mask, pts_live,
    method=cv2.RANSAC,
    ransacReprojThreshold=2.0,
    maxIters=2000,
    confidence=0.99
)

# Option B: Pure translation (comment above, uncomment below if needed)
# shifts = pts_live - pts_mask
# dx = np.median(shifts[:, 0])
# dy = np.median(shifts[:, 1])
# M_affine = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
# inliers = np.ones((len(good_matches), 1), dtype=np.uint8)

# === 10. Check transformation quality ===
if M_affine is None:
    print("❌ Error: Could not estimate transformation!")
    exit()

if inliers is not None:
    inlier_count = np.sum(inliers)
    inlier_ratio = inlier_count / len(good_matches)
    print(f"\nInliers: {inlier_count}/{len(good_matches)} ({inlier_ratio:.1%})")
    
    if inlier_ratio < 0.3:
        print("⚠️  Warning: Low inlier ratio - registration may be unreliable")
else:
    print("⚠️  Warning: No inlier information available")

# === 11. Extract transformation parameters ===
dx = M_affine[0, 2]
dy = M_affine[1, 2]
angle = np.arctan2(M_affine[1, 0], M_affine[0, 0]) * 180 / np.pi
scale_x = np.sqrt(M_affine[0, 0]**2 + M_affine[1, 0]**2)
scale_y = np.sqrt(M_affine[0, 1]**2 + M_affine[1, 1]**2)

print(f"\nTransformation Parameters:")
print(f"  Translation: dx = {dx:.2f} pixels, dy = {dy:.2f} pixels")
print(f"  Rotation: {angle:.2f} degrees")
print(f"  Scale: x = {scale_x:.3f}, y = {scale_y:.3f}")

print(f"\nAffine transformation matrix:")
print(M_affine)

# === 12. Visualize detected features ===
mask_vis = cv2.drawKeypoints(mask_frame, kp_mask, None, color=(0, 255, 0), flags=0)
live_vis = cv2.drawKeypoints(live_frame, kp_live, None, color=(0, 255, 0), flags=0)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title(f"Mask Frame {mask_frame_idx} - {len(kp_mask)} ORB Features")
plt.imshow(mask_vis, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Live Frame {live_frame_idx} - {len(kp_live)} ORB Features")
plt.imshow(live_vis, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# === 13. Visualize feature matches ===
matches_mask = inliers.ravel().tolist() if inliers is not None else None
match_vis = cv2.drawMatches(
    mask_frame, kp_mask,
    live_frame, kp_live,
    good_matches, None,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matches_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.title(f"Feature Matches (Green = Inliers, {inlier_count if inliers is not None else 'N/A'} matches)")
plt.imshow(match_vis)
plt.axis('off')
plt.tight_layout()
plt.show()

# === 14. Apply transformation to align mask frame ===
h, w = live_frame.shape
aligned_mask = cv2.warpAffine(mask_frame, M_affine, (w, h), flags=cv2.INTER_LINEAR)

# === 15. Visualize alignment result ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title(f"Original Mask (Frame {mask_frame_idx})")
plt.imshow(mask_frame, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Live Frame (Frame {live_frame_idx})")
plt.imshow(live_frame, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Aligned Mask")
plt.imshow(aligned_mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# === 16. Create difference images for quality assessment ===
# Before alignment
diff_before = cv2.absdiff(mask_frame, live_frame)

# After alignment
diff_after = cv2.absdiff(aligned_mask, live_frame)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Before Alignment")
plt.imshow(diff_before, cmap='hot')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("After Alignment")
plt.imshow(diff_after, cmap='hot')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Improvement")
improvement = diff_before.astype(float) - diff_after.astype(float)
plt.imshow(improvement, cmap='RdYlGn')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()

# === 17. Calculate alignment quality metrics ===
mse_before = np.mean((mask_frame.astype(float) - live_frame.astype(float))**2)
mse_after = np.mean((aligned_mask.astype(float) - live_frame.astype(float))**2)
improvement_pct = ((mse_before - mse_after) / mse_before) * 100

print(f"\nAlignment Quality:")
print(f"  MSE before: {mse_before:.2f}")
print(f"  MSE after: {mse_after:.2f}")
print(f"  Improvement: {improvement_pct:.1f}%")

# === 18. Optional: Overlay visualization ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Before Alignment Overlay")
overlay_before = cv2.addWeighted(mask_frame, 0.5, live_frame, 0.5, 0)
plt.imshow(overlay_before, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("After Alignment Overlay")
overlay_after = cv2.addWeighted(aligned_mask, 0.5, live_frame, 0.5, 0)
plt.imshow(overlay_after, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Checkerboard Comparison")
# Create checkerboard pattern
checker_size = 50
checker = np.indices((h, w)).sum(axis=0) // checker_size % 2
composite = np.where(checker, aligned_mask, live_frame)
plt.imshow(composite, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Processing complete!")
print(f"Transformation matrix saved in variable 'M_affine'")
print(f"Aligned mask saved in variable 'aligned_mask'")