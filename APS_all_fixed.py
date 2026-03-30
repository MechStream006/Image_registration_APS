"""
APS_all_fixed.py

Complete, debugged, and integrated script:
- Robust skeleton (ximgproc / skimage / morphological fallback)
- warp helpers (warp_image / ensure_homography)
- fast statistical vessel detection (local z-score + optional Wilcoxon refinement)
- gradient-weighted and entropy-based registration masks
- motion estimators (SIFT/ORB + RANSAC, phase-correlation, ECC wrapper, hybrid)
- blending fix (proper broadcasting)
- multi-frame DICOM loader (pydicom)
- visualization using matplotlib (frame, confidence map, skeleton overlay)
"""

import logging
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

try:
    import cv2
except Exception as exc:
    raise ImportError("OpenCV is required. Install with `pip install opencv-python`") from exc

try:
    import pydicom
except Exception as exc:
    raise ImportError("pydicom is required. Install with `pip install pydicom`") from exc

# Optional imports
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

# SIFT factory resolution (works with opencv-contrib)
try:
    if hasattr(cv2, "SIFT_create"):
        _SIFT_FACTORY = cv2.SIFT_create
    elif hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        _SIFT_FACTORY = cv2.xfeatures2d.SIFT_create
    else:
        _SIFT_FACTORY = None
except Exception:
    _SIFT_FACTORY = None

_has_ximgproc = hasattr(cv2, "ximgproc")

logger = logging.getLogger(__name__)


# ----------------------------- Basic helpers ---------------------------------

def ensure_gray_float(img: np.ndarray) -> np.ndarray:
    """Return single-channel float32 image normalized to [0,1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = img.astype(np.float32)
    if out.max() > 1.0:
        out /= 255.0
    return out


def ensure_homography(M: np.ndarray) -> np.ndarray:
    """Convert 2x3 affine to 3x3 homography or validate 3x3 matrix."""
    M = np.asarray(M, dtype=np.float32)
    if M.shape == (2, 3):
        H = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
        return H
    if M.shape == (3, 3):
        return M
    raise ValueError(f"Unexpected transform shape {M.shape}")


def warp_image(img: np.ndarray, M: np.ndarray, dsize=None,
               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT):
    """Warp image with either a 2x3 affine matrix or 3x3 homography."""
    if dsize is None:
        dsize = (img.shape[1], img.shape[0])
    M = np.asarray(M)
    if M.shape == (2, 3):
        return cv2.warpAffine(img, M, dsize, flags=flags, borderMode=borderMode)
    elif M.shape == (3, 3):
        return cv2.warpPerspective(img, M, dsize, flags=flags, borderMode=borderMode)
    else:
        raise ValueError(f"Unsupported transform shape: {M.shape}")


# ----------------------------- Skeleton helper -------------------------------

def compute_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
    """
    Compute a binary skeleton (uint8 0/255) from vessel_mask with fallbacks.
    Uses cv2.ximgproc.thinning when available, falls back to skimage.skeletonize,
    then to a morphological thinning approximation.
    """
    if vessel_mask.dtype != np.uint8:
        vessel_mask = ((vessel_mask > 0).astype(np.uint8) * 255)

    # Try OpenCV ximgproc thinning
    if _has_ximgproc:
        try:
            sk = cv2.ximgproc.thinning(vessel_mask)
            return (sk > 0).astype(np.uint8) * 255
        except Exception as e:
            logger.debug("cv2.ximgproc.thinning failed, falling back: %s", e)

    # Fall back to skimage
    if skeletonize is not None:
        try:
            sk_bool = skeletonize((vessel_mask > 0))
            return (sk_bool.astype(np.uint8) * 255)
        except Exception as e:
            logger.debug("skimage.skeletonize failed, falling back: %s", e)

    # Last resort: simple morphological thinning approximation
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        prev = vessel_mask.copy()
        sk = np.zeros_like(vessel_mask)
        for _ in range(20):
            eroded = cv2.erode(prev, kernel)
            temp = cv2.dilate(eroded, kernel)
            subset = cv2.subtract(prev, temp)
            sk = cv2.bitwise_or(sk, subset)
            prev = eroded
            if np.all(prev == 0):
                break
        return (sk > 0).astype(np.uint8) * 255
    except Exception as e:
        logger.exception("Fallback skeletonization failed: %s", e)
        return (vessel_mask > 0).astype(np.uint8) * 255


# ----------------------------- Mask helpers ---------------------------------

def _gradient_weighted_mask(image: np.ndarray, ksize: int = 3, threshold: float = 0.05) -> np.ndarray:
    """
    Returns a float mask in [0,1] where strong gradients get weight ~1 and smooth
    areas get weight ~0. Useful to emphasize vessel/edge regions for registration.
    """
    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax - mmin < 1e-9:
        return np.zeros_like(mag, dtype=np.float32)
    norm = (mag - mmin) / (mmax - mmin)
    mask = np.clip((norm - threshold) / (1.0 - threshold), 0.0, 1.0)
    return mask.astype(np.float32)


def _entropy_based_mask(image: np.ndarray, neighborhood: int = 9) -> np.ndarray:
    """
    Returns a mask (float 0..1) based on local variance as a cheap entropy proxy.
    """
    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    img = img.astype(np.float32)
    local_mean = ndi.uniform_filter(img, size=neighborhood)
    local_sqr = ndi.uniform_filter(img * img, size=neighborhood)
    local_var = np.maximum(local_sqr - local_mean * local_mean, 0.0)
    vmin, vmax = local_var.min(), local_var.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(local_var, dtype=np.float32)
    ent_proxy = (local_var - vmin) / (vmax - vmin)
    return ent_proxy.astype(np.float32)


def create_registration_mask_advanced(frame: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """
    Create a float mask in [0,1] for registration weighting.
    method: 'adaptive' (default), 'gradient_weighted', 'entropy_based'
    """
    method = method.lower()
    if method == 'adaptive':
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        var = ndi.uniform_filter(gray.astype(np.float32) ** 2, size=9) - ndi.uniform_filter(gray.astype(np.float32), size=9) ** 2
        vm = (var - var.min()) / (var.max() - var.min() + 1e-9)
        imn = (gray.astype(np.float32) - gray.min()) / (gray.max() - gray.min() + 1e-9)
        mask = 0.6 * vm + 0.4 * imn
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    elif method == 'gradient_weighted':
        return _gradient_weighted_mask(frame)
    elif method == 'entropy_based':
        return _entropy_based_mask(frame)
    else:
        logger.warning("Unknown mask method '%s' — falling back to adaptive.", method)
        return create_registration_mask_advanced(frame, method='adaptive')


# ------------------------ Statistical vessel detection -----------------------

def statistical_vessel_detection_fast(frames: np.ndarray, window: int = 15, z_thresh: float = 2.5,
                                      wilcoxon_on_candidates: bool = True, p_thresh: float = 0.05):
    """
    frames: ndarray (T, H, W) or (T, H, W, C) with grayscale or single-channel images.
    Returns a confidence map (H,W) scaled 0..1 (float32).
    - Uses local z-score (fast) to find candidate pixels where the current frame deviates
      from the local background across time.
    - Optionally applies Wilcoxon signed-rank test only to candidate pixels to refine p-values.
    """
    if frames.ndim == 4:
        # convert to grayscale luminance approx
        frames_gray = np.dot(frames[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        frames_gray = frames.astype(np.float32)

    T, H, W = frames_gray.shape
    if T < 2:
        return np.zeros((H, W), dtype=np.float32)

    current = frames_gray[-1].astype(np.float32)
    baseline = frames_gray[:-1]  # all frames except current

    # Local mean/std smoothing via uniform filter
    local_mean = ndi.uniform_filter(current.astype(np.float32), size=window)
    local_sqr_mean = ndi.uniform_filter((current.astype(np.float32) ** 2), size=window)
    local_var = np.maximum(local_sqr_mean - (local_mean ** 2), 1e-8)
    local_std = np.sqrt(local_var)

    # z-score map of current pixel relative to local distribution
    z_map = (current - local_mean) / (local_std + 1e-8)

    # Candidate mask using z threshold
    candidates = np.abs(z_map) >= z_thresh

    # Confidence map via smooth transform of z (not a p-value)
    conf = 1.0 - (1.0 / (1.0 + np.exp(np.abs(z_map) - z_thresh)))
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

    if wilcoxon_on_candidates:
        n_baseline = baseline.shape[0]
        if n_baseline > 0:
            baseline_flat = baseline.reshape(n_baseline, -1)  # (T-1, H*W)
            curr_flat = current.ravel()
            cand_idx = np.nonzero(candidates.ravel())[0]
            for idx in cand_idx:
                sample = baseline_flat[:, idx]
                try:
                    stat, pval = stats.wilcoxon(sample, np.full_like(sample, curr_flat[idx]))
                    if np.isnan(pval):
                        pval = 1.0
                    mapped = 1.0 - min(1.0, pval / max(1e-12, p_thresh))
                except Exception:
                    mapped = conf.ravel()[idx]
                conf.ravel()[idx] = float(mapped)

    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
    return conf


# ----------------------------- Motion estimation ----------------------------

def find_transform_ecc_safe(template: np.ndarray, target: np.ndarray, warp_mode=cv2.MOTION_AFFINE,
                            number_of_iterations=5000, termination_eps=1e-6, input_mask=None):
    """
    Safe wrapper around cv2.findTransformECC that ensures dtype and shape requirements.
    Returns (warp_matrix, success_flag, cc)
    """
    tpl = template
    tgt = target
    if tpl.ndim == 3:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    if tgt.ndim == 3:
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)

    img0 = tpl.astype(np.float32)
    img1 = tgt.astype(np.float32)

    if img0.max() > 1.0:
        img0 = img0 / 255.0
    if img1.max() > 1.0:
        img1 = img1 / 255.0

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    try:
        if input_mask is not None:
            cc, warp_matrix = cv2.findTransformECC(img0, img1, warp_matrix, warp_mode, criteria, inputMask=input_mask, gaussFiltSize=5)
        else:
            cc, warp_matrix = cv2.findTransformECC(img0, img1, warp_matrix, warp_mode, criteria, gaussFiltSize=5)
        return warp_matrix, True, cc
    except cv2.error as e:
        logger.warning("findTransformECC failed: %s", e)
        return warp_matrix, False, None


def _estimate_motion_sift_ransac(img0: np.ndarray, img1: np.ndarray, max_features=2000, ransac_thresh=3.0):
    """
    Estimate a homography using SIFT (or ORB fallback) + RANSAC.
    Returns (H (3x3), success_bool).
    """
    if img0.ndim == 3:
        g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g0, g1 = img0, img1

    if _SIFT_FACTORY is not None:
        detector = _SIFT_FACTORY()
    else:
        detector = cv2.ORB_create(nfeatures=max_features)

    kp0, des0 = detector.detectAndCompute(g0, None)
    kp1, des1 = detector.detectAndCompute(g1, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        logger.debug("Not enough features for SIFT/ORB matching.")
        return np.eye(3, dtype=np.float32), False

    if des0.dtype == np.uint8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des0, des1, k=2)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(des0, des1, k=2)

    good = []
    for m_n in raw_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        logger.debug("Not enough good matches: %d", len(good))
        return np.eye(3, dtype=np.float32), False

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thresh)
    if H is None:
        logger.debug("findHomography returned None")
        return np.eye(3, dtype=np.float32), False
    return H.astype(np.float32), True


def _estimate_motion_phase_correlation(img0: np.ndarray, img1: np.ndarray):
    """
    Estimate translation using phase correlation. Returns 2x3 affine that contains translation.
    """
    if img0.ndim == 3:
        g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g0, g1 = img0, img1

    g0f = g0.astype(np.float32)
    g1f = g1.astype(np.float32)
    hann = cv2.createHanningWindow((g0f.shape[1], g0f.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(g0f * hann, g1f * hann)
    dx, dy = shift
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    return M, True


def _estimate_motion_hybrid(img0: np.ndarray, img1: np.ndarray):
    """
    Try SIFT+RANSAC first, fallback to ECC affine, then to phase correlation.
    Returns warp (3x3 or 2x3) and bool success.
    """
    H, ok = _estimate_motion_sift_ransac(img0, img1)
    if ok:
        return H, True

    warp, ok_ecc, _ = find_transform_ecc_safe(img0, img1, warp_mode=cv2.MOTION_AFFINE)
    if ok_ecc:
        return warp, True

    M, ok_pc = _estimate_motion_phase_correlation(img0, img1)
    return M, ok_pc


# ------------------------------ Blending utils -------------------------------

def blend_warped_and_original(original: np.ndarray, warped: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Blend warped into original using weights map in [0,1].
    weights may be 2D (H,W). Handles color images with broadcasting.
    Returns same dtype as original.
    """
    if original.shape[:2] != weights.shape[:2]:
        raise ValueError("weights must match original image height/width")

    w = weights.astype(np.float32)
    if original.ndim == 2 or (original.ndim == 3 and original.shape[2] == 1):
        result = (w * warped + (1.0 - w) * original)
    else:
        w = w[..., np.newaxis]
        result = (w * warped + (1.0 - w) * original)

    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        result = np.clip(result, info.min, info.max)
        return result.astype(original.dtype)
    else:
        return result.astype(original.dtype)


# ----------------------------- DICOM Loader ---------------------------------

def load_dicom_loop(filepath: str) -> np.ndarray:
    """
    Load a multi-frame DICOM (cine/angiography) into a numpy array.

    Returns:
        frames: ndarray of shape (T, H, W), dtype=uint8
    """
    ds = pydicom.dcmread(filepath)
    if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
        arr = ds.pixel_array  # shape (T, H, W)
    else:
        arr = ds.pixel_array[np.newaxis, ...]

    arr = np.array(arr)

    # Normalize scale to 0..255 uint8 for processing and visualization
    maxv = float(arr.max()) if arr.size > 0 else 1.0
    if maxv <= 0:
        maxv = 1.0
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) / maxv * 255.0).astype(np.uint8)

    return arr


# ------------------------------- Diagnostics --------------------------------

def quick_summary(frames: np.ndarray) -> dict:
    frames = np.asarray(frames)
    summary = {
        'shape': frames.shape,
        'dtype': str(frames.dtype),
        'min': float(frames.min()) if frames.size else None,
        'max': float(frames.max()) if frames.size else None,
        'mean': float(frames.mean()) if frames.size else None
    }
    return summary


# ------------------------------- Demo / Visualization ------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # ====== Edit this path to your multi-frame DICOM cine file ======
    dicom_path = r"C:\path\to\your\cine_file.dcm"   # <-- REPLACE with your path

    try:
        frames = load_dicom_loop(dicom_path)
        print("Loaded DICOM loop:", frames.shape, frames.dtype)
        print("Summary:", quick_summary(frames))

        # Run fast vessel detection (last frame vs baseline)
        conf_map = statistical_vessel_detection_fast(frames, window=15, z_thresh=2.5, wilcoxon_on_candidates=True)
        print("Confidence map shape:", conf_map.shape, "min/max:", conf_map.min(), conf_map.max())

        # Create registration mask for first frame
        mask = create_registration_mask_advanced(frames[0], method="adaptive")
        print("Registration mask stats:", mask.min(), mask.max())

        # Skeletonize detected vessels (threshold at 0.5)
        vessel_mask = (conf_map > 0.5).astype(np.uint8) * 255
        skeleton = compute_skeleton(vessel_mask)
        print("Skeleton pixel count:", int(np.count_nonzero(skeleton)))

        # ---------------- VISUALIZATION ----------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original first frame
        axes[0].imshow(frames[0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("First Frame (from DICOM)")
        axes[0].axis("off")

        # Confidence map
        im1 = axes[1].imshow(conf_map, cmap="hot")
        axes[1].set_title("Vessel Confidence Map")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Skeleton overlay on first frame (red)
        overlay = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
        overlay[skeleton > 0] = [255, 0, 0]  # red skeleton
        axes[2].imshow(overlay[..., ::-1] if overlay.shape[-1] == 3 else overlay, vmin=0, vmax=255)  # convert BGR->RGB if needed
        axes[2].set_title("Skeleton Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("Error loading or processing DICOM: %s", e)
        print("Error loading or processing DICOM:", e)

    print("Demo finished.")
