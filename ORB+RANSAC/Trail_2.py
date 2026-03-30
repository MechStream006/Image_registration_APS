import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === 1. Read DICOM cine loop ===
# Replace this with your DICOM file path
dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
ds = pydicom.dcmread(dicom_path)

# DICOM cine loops store frames in a 3D pixel array (frames × rows × cols)
frames = ds.pixel_array.astype(np.uint8)

print(f"Total frames: {frames.shape[0]}")
print(f"Frame size: {frames.shape[1]} × {frames.shape[2]}")

# === 2. Select mask and live frames ===
mask_frame = frames[0]           # first frame
live_frame = frames[10]           # e.g., second frame (you can change index)

# Normalize if pixel depth > 8 bits
if mask_frame.max() > 255:
    mask_frame = cv2.convertScaleAbs(mask_frame, alpha=(255.0 / mask_frame.max()))
    live_frame = cv2.convertScaleAbs(live_frame, alpha=(255.0 / live_frame.max()))

# === 3. Apply ORB Feature Detector ===
orb = cv2.ORB_create(nfeatures=2000)

# Detect keypoints and descriptors
kp_mask, des_mask = orb.detectAndCompute(mask_frame, None)
kp_live, des_live = orb.detectAndCompute(live_frame, None)

print(f"Mask frame: {len(kp_mask)} keypoints")
print(f"Live frame: {len(kp_live)} keypoints")

# === 4. Visualize detected features ===
mask_vis = cv2.drawKeypoints(mask_frame, kp_mask, None, color=(0,255,0), flags=0)
live_vis = cv2.drawKeypoints(live_frame, kp_live, None, color=(0,255,0), flags=0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des_mask, des_live, k=2)

# --- 2️⃣ Apply ratio test to keep only strong matches ---
good_matches = []
for m, n in matches:
    if m.distance < 0.90 * n.distance:
        good_matches.append(m)

print(f"Total matches: {len(matches)},  Good matches after ratio test: {len(good_matches)}")

# --- 3️⃣ Extract matched keypoint coordinates ---
pts_mask = np.float32([kp_mask[m.queryIdx].pt for m in good_matches])
pts_live = np.float32([kp_live[m.trainIdx].pt for m in good_matches])

# --- 4️⃣ Estimate affine transformation using RANSAC ---
M, inliers = cv2.estimateAffinePartial2D(
    pts_mask, pts_live,
    method=cv2.RANSAC,
    ransacReprojThreshold=2.0
)

print("Affine transformation matrix:\n", M)

# --- 5️⃣ Visualize inlier matches ---
matches_mask = inliers.ravel().tolist() if inliers is not None else None
match_vis = cv2.drawMatches(
    mask_frame, kp_mask,
    live_frame, kp_live,
    good_matches, None,
    matchColor=(0,255,0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=2
)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Mask Frame - ORB Features")
plt.imshow(mask_vis, cmap='gray')

plt.subplot(1,2,2)
plt.title("Live Frame - ORB Features")
plt.imshow(live_vis, cmap='gray')
plt.show()

# --- 5️⃣ Visualize inlier matches ---
matches_mask = inliers.ravel().tolist() if inliers is not None else None
match_vis = cv2.drawMatches(
    mask_frame, kp_mask,
    live_frame, kp_live,
    good_matches, None,
    matchColor=(0,255,0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=2
)

plt.figure(figsize=(14, 6))
plt.title("Feature Matches (Green = Inliers)")
plt.imshow(match_vis)
plt.show()

# --- 6️⃣ Warp mask frame using the estimated affine transform ---
h, w = live_frame.shape
aligned_mask = cv2.warpAffine(mask_frame, M, (w, h))

# --- 7️⃣ Visualize alignment result ---
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Original Mask")
plt.imshow(mask_frame, cmap='gray')

plt.subplot(1,3,2)
plt.title("Live Frame")
plt.imshow(live_frame, cmap='gray')

plt.subplot(1,3,3)
plt.title("Aligned Mask (after affine)")
plt.imshow(aligned_mask, cmap='gray')

plt.tight_layout()
plt.show()