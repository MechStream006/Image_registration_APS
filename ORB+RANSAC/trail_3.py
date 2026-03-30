# Save this as evaluate_nfeatures.py and run it in an environment with OpenCV, pydicom, matplotlib, numpy, pandas installed.
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error

def read_dicom_frames(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    frames = ds.pixel_array.astype(np.float32)
    # scale to 0..255 if >8bit
    if frames.max() > 255:
        frames = (255.0 * (frames - frames.min()) / (frames.max() - frames.min())).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)
    return frames

def orb_detect_and_describe(img, nfeatures):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, des = orb.detectAndCompute(img, None)
    return kps, des

def match_descriptors(des1, des2, ratio_thresh=0.75):
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in raw_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good

def ransac_translation(kp1, kp2, matches, residual_threshold=3.0):
    if len(matches) < 3:
        return None, [], np.inf
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    # Solve affine (or translation only). We'll estimate an affine using cv2.estimateAffinePartial2D (RANSAC).
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=residual_threshold)
    if inliers is None:
        inliers = []
    else:
        inliers = inliers.ravel().astype(bool)
    # compute reprojection error for inliers
    if M is not None and np.any(inliers):
        pts1_h = np.hstack([pts1[inliers], np.ones((np.sum(inliers), 1))])  # Nx3
        projected = (M @ pts1_h.T).T  # Nx2
        err = np.sqrt(np.sum((projected - pts2[inliers])**2, axis=1))
        median_err = np.median(err)
    else:
        median_err = np.inf
    return M, inliers, median_err

def compute_coverage(kps, img_shape, grid_size=64):
    # Fraction of grid cells that contain at least one keypoint (coarse coverage measure)
    h, w = img_shape
    grid_h = (h + grid_size - 1) // grid_size
    grid_w = (w + grid_size - 1) // grid_size
    occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        gx = min(x // grid_size, grid_w - 1)
        gy = min(y // grid_size, grid_h - 1)
        occupancy[gy, gx] = 1
    return occupancy.mean()  # fraction occupied

def evaluate_on_pair(mask_img, live_img, nfeatures_list=[100,500,1000,2000,3000,5000,10000]):
    results = []
    for nf in nfeatures_list:
        t0 = time.time()
        kps_m, des_m = orb_detect_and_describe(mask_img, nf)
        kps_l, des_l = orb_detect_and_describe(live_img, nf)
        detect_time = time.time() - t0

        t1 = time.time()
        matches = match_descriptors(des_m, des_l, ratio_thresh=0.75)
        match_time = time.time() - t1

        M, inliers_mask, median_err = ransac_translation(kps_m, kps_l, matches, residual_threshold=3.0)
        inlier_count = int(np.sum(inliers_mask)) if len(inliers_mask) else 0
        match_count = len(matches)
        kp_m_count = len(kps_m)
        kp_l_count = len(kps_l)
        inlier_ratio = inlier_count / match_count if match_count > 0 else 0.0

        coverage_m = compute_coverage(kps_m, mask_img.shape, grid_size=64)
        coverage_l = compute_coverage(kps_l, live_img.shape, grid_size=64)

        results.append({
            'nfeatures': nf,
            'kp_mask': kp_m_count,
            'kp_live': kp_l_count,
            'matches': match_count,
            'inliers': inlier_count,
            'inlier_ratio': inlier_ratio,
            'median_reproj_err': median_err,
            'detect_time_s': detect_time,
            'match_time_s': match_time,
            'coverage_mask': coverage_m,
            'coverage_live': coverage_l
        })
        print(f"nfeatures={nf:5d}  kp_mask={kp_m_count:4d}  kp_live={kp_l_count:4d}  matches={match_count:4d}  inliers={inlier_count:4d}  inlier_ratio={inlier_ratio:.3f}  median_err={median_err:.2f}  times={detect_time+match_time:.2f}s")
    return pd.DataFrame(results)

# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"   # <- replace with your path
    frames = read_dicom_frames(dicom_path)
    mask = frames[0]
    live = frames[14]   # you can loop other frames / evaluate all frames for statistics

    nfeatures_list = [100, 500, 1000, 2000, 3000, 5000, 10000]
    df = evaluate_on_pair(mask, live, nfeatures_list=nfeatures_list)

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.ravel()
    df.plot(x='nfeatures', y=['kp_mask','kp_live'], marker='o', ax=axs[0], title='Keypoints count')
    df.plot(x='nfeatures', y='matches', marker='o', ax=axs[1], title='Matches (ratio test)')
    df.plot(x='nfeatures', y='inliers', marker='o', ax=axs[2], title='Inliers (RANSAC)')
    df.plot(x='nfeatures', y='inlier_ratio', marker='o', ax=axs[3], title='Inlier ratio')
    plt.tight_layout()
    plt.show()

    df.to_csv("nfeatures_evaluation.csv", index=False)
    print("\nSaved nfeatures_evaluation.csv")
