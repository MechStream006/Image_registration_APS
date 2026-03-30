import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import cv2
import pydicom
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Logging utilities
# ---------------------------

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'process.log')

    logger = logging.getLogger('ecc_warp')
    logger.setLevel(logging.DEBUG)
    # Prevent adding multiple handlers in interactive runs
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug(f"Logger initialized, file={log_file}")
    return logger


# ---------------------------
# IO utilities (DICOM / images)
# ---------------------------

def read_dicom_frames(path: str) -> np.ndarray:
    """Read DICOM and return frames as float32 array (n_frames, H, W) or (n_frames, H, W, C)"""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.float32)


def read_image_as_frames(path: str) -> np.ndarray:
    """Read a single image file and return as a 1-frame ndarray."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    # convert to float32
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    if im.ndim == 2:
        im = im[np.newaxis, ...]
    else:
        im = im[np.newaxis, ...]
    return im


def read_frames_auto(input_path: str) -> np.ndarray:
    """Auto-detect DICOM (.dcm) or image file. Returns numpy array of frames."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if os.path.isdir(input_path):
        # read all images in directory sorted
        files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)])
        frames = []
        for f in files:
            try:
                im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                frames.append(im.astype(np.float32))
            except Exception:
                continue
        if not frames:
            raise RuntimeError(f"No readable images in directory: {input_path}")
        arr = np.stack(frames, axis=0)
        return arr
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.dcm':
        return read_dicom_frames(input_path)
    else:
        return read_image_as_frames(input_path)


def save_image(path: str, image: np.ndarray) -> None:
    """Save image, normalizing floats to 0-255 if needed."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # clip then scale if values are in [0,1] or any range
        imin, imax = np.min(image), np.max(image)
        if imax <= 1.0 and imin >= 0.0:
            img8 = (image * 255.0).astype(np.uint8)
        else:
            # rescale to 0..255
            if imax - imin < 1e-9:
                img8 = np.zeros_like(image, dtype=np.uint8)
            else:
                img8 = np.clip((image - imin) / (imax - imin) * 255.0, 0, 255).astype(np.uint8)
    else:
        img8 = image.astype(np.uint8)
    # ensure directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img8)


# ---------------------------
# Image helpers
# ---------------------------

def normalize_to_float32(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx - mn < 1e-9:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - mn) / (mx - mn)
    return out.astype(np.float32)


def prepare_for_ecc(img: np.ndarray) -> np.ndarray:
    """Make single-channel float32 image in [0,1] for ECC."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2GRAY).astype(np.float32)
    elif img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        # unknown shape, try to collapse channels
        gray = np.mean(img, axis=2).astype(np.float32)
    return normalize_to_float32(gray)


# ---------------------------
# Decomposition and utils
# ---------------------------

def decompose_warp2d(warp: np.ndarray, motion: int) -> Dict[str, Any]:
    """Approximate decomposition of 2D warp (translation, rotation_deg, scales, shear)."""
    try:
        if motion == cv2.MOTION_TRANSLATION:
            tx, ty = float(warp[0,2]), float(warp[1,2])
            return dict(translation_x=tx, translation_y=ty, rotation_deg=0.0, scale_x=1.0, scale_y=1.0, shear=0.0)
        elif motion == cv2.MOTION_EUCLIDEAN:
            tx, ty = float(warp[0,2]), float(warp[1,2])
            angle_rad = np.arctan2(warp[1,0], warp[0,0])
            return dict(translation_x=tx, translation_y=ty, rotation_deg=float(np.degrees(angle_rad)), scale_x=1.0, scale_y=1.0, shear=0.0)
        elif motion == cv2.MOTION_AFFINE:
            A = warp[:,:2].astype(np.float64)
            tx, ty = float(warp[0,2]), float(warp[1,2])
            sx = float(np.linalg.norm(A[:,0]))
            sy = float(np.linalg.norm(A[:,1]))
            theta = float(np.degrees(np.arctan2(A[1,0], A[0,0])))
            shear = float(np.dot(A[:,0], A[:,1]) / (sx * sy)) if sx * sy > 1e-9 else 0.0
            return dict(translation_x=tx, translation_y=ty, rotation_deg=theta, scale_x=sx, scale_y=sy, shear=shear)
        elif motion == cv2.MOTION_HOMOGRAPHY:
            tx, ty = float(warp[0,2]), float(warp[1,2])
            return dict(translation_x=tx, translation_y=ty, rotation_deg='N/A', scale_x='N/A', scale_y='N/A', shear='N/A')
    except Exception:
        return dict(translation_x=0, translation_y=0, rotation_deg=0, scale_x=1, scale_y=1, shear=0)


# ---------------------------
# Warp / remap helpers
# ---------------------------

def warp_frame(frame: np.ndarray, warp: np.ndarray, motion_model: int) -> np.ndarray:
    """Warp `frame` using `warp` (2x3 or 3x3). Returns same dtype as frame (float32 or uint8)."""
    h, w = frame.shape[:2]
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(frame, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        warped = cv2.warpAffine(frame, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return warped


def ensure_mask(mask: Optional[np.ndarray], height: int, width: int) -> np.ndarray:
    """Return binary mask in float32 in shape (H,W) with values 0..1."""
    if mask is None:
        return np.ones((height, width), dtype=np.float32)
    if mask.ndim == 3:
        # convert to gray
        m = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        m = mask
    m = m.astype(np.float32)
    # normalize to 0..1
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    return m


# ---------------------------
# Global ECC estimation (multi-scale)
# ---------------------------

def estimate_motion_ecc(template: np.ndarray, image: np.ndarray,
                        motion=cv2.MOTION_AFFINE,
                        number_of_iterations: int = 200,
                        termination_eps: float = 1e-6,
                        gaussFiltSize: int = 1,
                        pyr_levels: int = 0,
                        logger: Optional[logging.Logger] = None,
                        mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """
    Multi-scale ECC estimation. Returns (ecc_val, warp_matrix)
    Robust handling of different OpenCV return signatures.
    """
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    T_full = prepare_for_ecc(template)
    I_full = prepare_for_ecc(image)
    h, w = T_full.shape[:2]

    # initial warp (full-resolution coordinates)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    logger.info(f"Starting ECC multi-scale: pyr_levels={pyr_levels}, motion={motion}")

    # iterate from coarse to fine
    for level in range(pyr_levels, -1, -1):
        scale = 1.0 / (2 ** level)
        Tw = cv2.resize(T_full, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        Iw = cv2.resize(I_full, (Tw.shape[1], Tw.shape[0]), interpolation=cv2.INTER_AREA)

        # prepare initial warp for this level: scale translations
        warp_lvl = warp.copy().astype(np.float32)
        if motion != cv2.MOTION_HOMOGRAPHY:
            # warp_lvl is 2x3
            warp_lvl[0, 2] = warp_lvl[0, 2] * scale
            warp_lvl[1, 2] = warp_lvl[1, 2] * scale
        else:
            warp_lvl[0, 2] = warp_lvl[0, 2] * scale
            warp_lvl[1, 2] = warp_lvl[1, 2] * scale

        logger.debug(f"Level {level}: scale={scale}, image size={Iw.shape}, init warp (lvl coords):\n{warp_lvl}")

        # build termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        try:
            # findTransformECC expects single-channel float images in [0,1]
            # OpenCV may accept inputMask to restrict area of computation (optional)
            mask_lvl = None
            if mask is not None:
                mask_f = ensure_mask(mask, h, w)
                mask_lvl = cv2.resize(mask_f, (Iw.shape[1], Iw.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_lvl = (mask_lvl > 0.5).astype(np.uint8)

            res = cv2.findTransformECC(Tw, Iw, warp_lvl, motion, criteria, mask_lvl, gaussFiltSize)
            if isinstance(res, tuple) or isinstance(res, list):
                if len(res) == 2:
                    ecc_val, warp_lvl_est = float(res[0]), res[1].astype(np.float32)
                else:
                    # older versions: (retval, warp)
                    ecc_val = float(res[0])
                    warp_lvl_est = res[1].astype(np.float32)
            else:
                ecc_val = float(res)
                # warp is updated in-place in some versions
                warp_lvl_est = warp_lvl

            logger.info(f"Level {level} ECC converged: ecc={ecc_val:.6f}")
            logger.debug(f"Warp (level {level}) result:\n{warp_lvl_est}")

            # map warp_lvl_est back to full-resolution warp
            warp = warp_lvl_est.copy().astype(np.float32)
            if level > 0:
                # prepare warp for next (finer) level by scaling translation by 2
                if motion != cv2.MOTION_HOMOGRAPHY:
                    warp[0, 2] = warp[0, 2] * 2.0
                    warp[1, 2] = warp[1, 2] * 2.0
                else:
                    warp[0, 2] = warp[0, 2] * 2.0
                    warp[1, 2] = warp[1, 2] * 2.0

        except cv2.error as e:
            logger.warning(f"ECC failed at level {level}: {e}")
            ecc_val = -1.0
            # keep current warp (identity or previous)
            # do not raise; allow caller to decide fallback

    logger.info("ECC multi-scale finished")
    return ecc_val, warp


# ---------------------------
# Global pipeline
# ---------------------------

def process_global_mode(frames: np.ndarray, mask_img: Optional[np.ndarray], output_dir: str,
                        motion_model=cv2.MOTION_AFFINE, pyr_levels: int = 1,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Process frames in global ECC mode: compute one warp per-frame, warp, and apply mask."""
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    os.makedirs(output_dir, exist_ok=True)
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])

    # Prepare template (frame 0)
    template_raw = frames[0]
    template = prepare_for_ecc(template_raw)

    mask = None
    if mask_img is not None:
        mask = ensure_mask(mask_img, h, w)
    else:
        mask = np.ones((h, w), dtype=np.float32)

    results = []
    csv_rows = []

    for idx in range(frames.shape[0]):
        logger.info(f"=== Processing frame {idx} (global) ===")
        live_raw = frames[idx]
        live_prepared = prepare_for_ecc(live_raw)

        # estimate
        ecc_val, warp = estimate_motion_ecc(template_raw, live_raw, motion=motion_model,
                                            pyr_levels=pyr_levels, logger=logger, mask=mask_img)

        params = decompose_warp2d(warp, motion_model)
        logger.debug(f"Decomposed warp params: {params}")

        # warp live frame to template coordinates
        try:
            warped_full = warp_frame(live_raw, warp, motion_model)
            logger.info("Warp applied to frame")
        except Exception as e:
            logger.exception(f"Failed to warp frame {idx}: {e}")
            warped_full = live_raw.copy()

        # Compose masked overlay: inside mask use warped pixel, outside use template
        # Work with color if available
        if live_raw.ndim == 2:
            warped_rgb = warped_full
            template_rgb = template_raw
        else:
            warped_rgb = warped_full
            template_rgb = template_raw

        mask_f = mask
        # Expand mask to channels if necessary
        if warped_rgb.ndim == 3 and mask_f.ndim == 2:
            mask3 = np.repeat(mask_f[:, :, None], warped_rgb.shape[2], axis=2)
        else:
            mask3 = mask_f

        composite = (warped_rgb.astype(np.float32) * mask3 + template_rgb.astype(np.float32) * (1.0 - mask3))

        # Save outputs and logs
        frame_base = os.path.join(output_dir, f'frame_{idx:04d}')
        save_image(frame_base + '_live.png', live_raw)
        save_image(frame_base + '_warped.png', warped_full)
        save_image(frame_base + '_composite.png', composite)

        # Compute difference images for diagnostics
        try:
            diff = np.abs(prepare_for_ecc(template_rgb) - prepare_for_ecc(warped_full))
            save_image(frame_base + '_diff.png', diff)
        except Exception:
            pass

        row = dict(frame_idx=int(idx), ecc=float(ecc_val), motion=int(motion_model))
        row.update(params)
        results.append(row)
        csv_rows.append(row)

        logger.info(f"Frame {idx}: ecc={ecc_val:.6f}, params={params}")

    df = pd.DataFrame(csv_rows)
    csv_out = os.path.join(output_dir, 'global_results.csv')
    df.to_csv(csv_out, index=False)
    logger.info(f"Saved global results CSV: {csv_out}")

    return dict(results=results, output_dir=output_dir)


# ---------------------------
# Grid ECC pipeline (per-cell)
# ---------------------------

def create_grid_coordinates(height: int, width: int, grid_rows: int, grid_cols: int) -> List[Dict[str, Any]]:
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
                'row': row, 'col': col, 'grid_id': f'R{row}C{col}',
                'y_start': y_start, 'y_end': y_end, 'x_start': x_start, 'x_end': x_end,
                'height': y_end - y_start, 'width': x_end - x_start
            })
    return grid_coords


def estimate_motion_ecc_grid(template_cell: np.ndarray, frame_cell: np.ndarray, motion=cv2.MOTION_AFFINE,
                             number_of_iterations: int = 200, termination_eps: float = 1e-6,
                             gaussFiltSize: int = 1, min_cell_size: int = 12,
                             logger: Optional[logging.Logger] = None) -> Tuple[float, Optional[np.ndarray], bool]:
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    if template_cell.shape[0] < min_cell_size or template_cell.shape[1] < min_cell_size:
        return -1.0, None, False

    T = prepare_for_ecc(template_cell)
    I = prepare_for_ecc(frame_cell)

    if np.std(T) < 0.01 or np.std(I) < 0.01:
        return -1.0, None, False

    h, w = T.shape[:2]
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    try:
        res = cv2.findTransformECC(T, I, warp, motion, criteria, None, gaussFiltSize)
        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) == 2:
                ecc_val, warp = float(res[0]), res[1].astype(np.float32)
            else:
                ecc_val = float(res[0])
        else:
            ecc_val = float(res)
        return float(ecc_val), warp, True
    except cv2.error as e:
        logger.debug(f"Cell ECC failed: {e}")
        return -1.0, None, False


def assemble_warped_frame_from_grid(live_raw: np.ndarray, grid_results: List[Dict[str, Any]],
                                    motion_model: int, mask: np.ndarray,
                                    blend_radius: int = 8, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Assemble a warped frame from per-cell warps. Uses accumulation & blended weights to avoid seams.
    Only places warped pixels where mask==1 (or mask>0.5). Returns composed frame (same shape as live_raw).
    """
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    H, W = live_raw.shape[:2]
    nch = 1 if live_raw.ndim == 2 else live_raw.shape[2]

    accum = np.zeros((H, W, nch), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for res in grid_results:
        y0, y1, x0, x1 = res['y_start'], res['y_end'], res['x_start'], res['x_end']
        cell_live = live_raw[y0:y1, x0:x1]
        cell_mask = mask[y0:y1, x0:x1]

        if not res['success'] or res['warp_matrix'] is None:
            logger.debug(f"Skipping cell {res['grid_id']} (no warp).")
            continue

        warp = np.array(res['warp_matrix'], dtype=np.float32)
        try:
            warped_cell = warp_frame(cell_live, warp, motion_model)
        except Exception as e:
            logger.debug(f"Warp failed for cell {res['grid_id']}: {e}")
            continue

        # create blending weight for this cell (smooth taper to edges)
        ch = cell_mask.shape
        yy = np.linspace(-1, 1, ch[0])[:, None]
        xx = np.linspace(-1, 1, ch[1])[None, :]
        # radial weight: higher in center, lower near border
        wcell = np.exp(-((yy**2 + xx**2) * (blend_radius / max(ch[0], ch[1]))))
        # combine with the binary mask
        wcell = (cell_mask > 0.5).astype(np.float32) * wcell

        if nch == 1:
            accum[y0:y1, x0:x1, 0] += (warped_cell.astype(np.float32) * wcell)
        else:
            accum[y0:y1, x0:x1, :] += (warped_cell.astype(np.float32) * wcell[:, :, None])
        weight[y0:y1, x0:x1] += wcell

        logger.debug(f"Added warped contribution for cell {res['grid_id']} (sum w={wcell.sum():.2f})")

    # avoid division by zero
    eps = 1e-6
    composed = np.zeros_like(accum)
    mask_nonzero = weight >= eps
    for c in range(nch):
        composed[..., c] = np.where(mask_nonzero, accum[..., c] / (weight + eps), 0.0)

    # Where we have no contributions (weight==0) fall back to original template (live_raw)
    composed_image = composed.astype(np.float32)
    if nch == 1:
        out = np.zeros((H, W), dtype=np.float32)
        out[mask_nonzero] = composed_image[..., 0][mask_nonzero]
        out[~mask_nonzero] = live_raw.astype(np.float32)[~mask_nonzero]
    else:
        out = live_raw.astype(np.float32).copy()
        out[mask_nonzero, :] = composed_image[mask_nonzero, :]

    return out


def analyze_frame_grid_motion(template_frame: np.ndarray, current_frame: np.ndarray, grid_rows: int, grid_cols: int,
                              motion_model: int, frame_idx: int, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    h, w = template_frame.shape[:2]
    grid_coords = create_grid_coordinates(h, w, grid_rows, grid_cols)
    results = []

    for grid_info in grid_coords:
        tcell = template_frame[grid_info['y_start']:grid_info['y_end'], grid_info['x_start']:grid_info['x_end']]
        icell = current_frame[grid_info['y_start']:grid_info['y_end'], grid_info['x_start']:grid_info['x_end']]

        if frame_idx == 0:
            ecc = 1.0
            if motion_model == cv2.MOTION_HOMOGRAPHY:
                warp = np.eye(3, dtype=np.float32)
            else:
                warp = np.eye(2, 3, dtype=np.float32)
            params = dict(translation_x=0, translation_y=0, rotation_deg=0.0, scale_x=1.0, scale_y=1.0, shear=0.0)
            success = True
        else:
            ecc, warp, success = estimate_motion_ecc_grid(tcell, icell, motion=motion_model, logger=logger)
            if success and warp is not None:
                params = decompose_warp2d(warp, motion_model)
            else:
                params = dict(translation_x=0.0, translation_y=0.0, rotation_deg=0.0, scale_x=1.0, scale_y=1.0, shear=0.0)

        results.append({
            'frame_idx': int(frame_idx), 'grid_id': grid_info['grid_id'], 'grid_row': grid_info['row'], 'grid_col': grid_info['col'],
            'x_start': grid_info['x_start'], 'y_start': grid_info['y_start'], 'x_end': grid_info['x_end'], 'y_end': grid_info['y_end'],
            'cell_width': grid_info['width'], 'cell_height': grid_info['height'],
            'ecc_correlation': float(ecc), 'success': bool(success),
            'translation_x': params['translation_x'], 'translation_y': params['translation_y'], 'rotation_deg': params['rotation_deg'],
            'scale_x': params['scale_x'], 'scale_y': params['scale_y'], 'shear': params['shear'],
            'warp_matrix': warp.tolist() if warp is not None else None
        })

    return results


def process_grid_mode(frames: np.ndarray, mask_img: Optional[np.ndarray], output_dir: str,
                      grid_rows: int = 8, grid_cols: int = 8, motion_model=cv2.MOTION_AFFINE,
                      logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger('ecc_warp')

    os.makedirs(output_dir, exist_ok=True)
    H, W = frames[0].shape[:2]

    template_raw = frames[0]
    template = prepare_for_ecc(template_raw)

    mask = None
    if mask_img is not None:
        mask = ensure_mask(mask_img, H, W)
    else:
        mask = np.ones((H, W), dtype=np.float32)

    all_results = []
    image_paths = []

    for idx in range(frames.shape[0]):
        logger.info(f"=== Processing frame {idx} (grid {grid_rows}x{grid_cols}) ===")
        cur_raw = frames[idx]
        cur_prepared = prepare_for_ecc(cur_raw)

        grid_results = analyze_frame_grid_motion(template_raw, cur_raw, grid_rows, grid_cols, motion_model, idx, logger=logger)
        all_results.append(grid_results)

        # Compose warped frame from grid warps and mask
        try:
            composed = assemble_warped_frame_from_grid(cur_raw, grid_results, motion_model, mask, logger=logger)
            frame_base = os.path.join(output_dir, f'frame_{idx:04d}')
            save_image(frame_base + '_live.png', cur_raw)
            save_image(frame_base + '_composed.png', composed)
            image_paths.append((idx, frame_base + '_composed.png'))
            logger.info(f"Saved composed grid-warp image for frame {idx}")
        except Exception as e:
            logger.exception(f"Failed to assemble composed image for frame {idx}: {e}")
            image_paths.append((idx, ''))

    # Flatten results to CSV
    flat = []
    for frame_results in all_results:
        flat.extend(frame_results)

    df = pd.DataFrame(flat)
    csv_out = os.path.join(output_dir, 'grid_results.csv')
    df.to_csv(csv_out, index=False)
    logger.info(f"Saved grid results CSV: {csv_out}")

    return dict(results=all_results, image_paths=image_paths, output_dir=output_dir)


# ---------------------------
# Visualization utilities (heatmaps)
# ---------------------------

def create_grid_visualization(template_frame: np.ndarray, current_frame: np.ndarray, grid_results: List[Dict[str, Any]],
                              frame_idx: int, grid_rows: int, grid_cols: int, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ecc_matrix = np.zeros((grid_rows, grid_cols))
    tx = np.zeros((grid_rows, grid_cols))
    ty = np.zeros((grid_rows, grid_cols))
    mag = np.zeros((grid_rows, grid_cols))

    for r in grid_results:
        i = r['grid_row']
        j = r['grid_col']
        ecc_matrix[i, j] = r['ecc_correlation'] if r['success'] else -1
        tx[i, j] = r['translation_x']
        ty[i, j] = r['translation_y']
        mag[i, j] = np.sqrt(float(r['translation_x'])**2 + float(r['translation_y'])**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Grid Motion Analysis - Frame {frame_idx}', fontsize=14)

    ax = axes[0, 0]
    ax.imshow(prepare_for_ecc(template_frame), cmap='gray')
    ax.set_title('Template (Frame 0)')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(prepare_for_ecc(current_frame), cmap='gray')
    ax.set_title(f'Current Frame ({frame_idx})')
    ax.axis('off')

    ax = axes[0, 2]
    im = ax.imshow(ecc_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('ECC correlation')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im2 = ax.imshow(tx, cmap='RdBu_r', vmin=-np.max(np.abs(tx)), vmax=np.max(np.abs(tx)))
    ax.set_title('Translation X')
    plt.colorbar(im2, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im3 = ax.imshow(ty, cmap='RdBu_r', vmin=-np.max(np.abs(ty)), vmax=np.max(np.abs(ty)))
    ax.set_title('Translation Y')
    plt.colorbar(im3, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    im4 = ax.imshow(mag, cmap='plasma')
    ax.set_title('Translation magnitude')
    plt.colorbar(im4, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


# ---------------------------
# Command-line interface
# ---------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Global + Grid ECC warp & masked pixel transfer')
    parser.add_argument('--mode', choices=['global', 'grid'], default='grid', help='Operation mode')
    parser.add_argument('--input', required=True, help='Input DICOM file or image or folder of images')
    parser.add_argument('--mask', required=False, help='Mask image file (optional). White==use region')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--motion', choices=['translation', 'euclidean', 'affine', 'homography'], default='affine')
    parser.add_argument('--pyr', type=int, default=1, help='Pyramid levels for global ECC')
    parser.add_argument('--grid', nargs=2, type=int, default=[8, 8], help='Grid rows cols for grid mode')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    # Setup
    outdir = args.output
    logger = setup_logger(outdir)
    logger.info('Starting ECC warp pipeline')

    frames = read_frames_auto(args.input)
    logger.info(f'Loaded frames: {frames.shape}')

    mask_img = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if m is None:
            logger.error(f'Could not load mask: {args.mask}')
            return
        mask_img = m
        logger.info(f'Loaded mask: {args.mask} shape={mask_img.shape}')

    motion_map = dict(translation=cv2.MOTION_TRANSLATION, euclidean=cv2.MOTION_EUCLIDEAN,
                      affine=cv2.MOTION_AFFINE, homography=cv2.MOTION_HOMOGRAPHY)
    motion_model = motion_map[args.motion]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_out = os.path.join(outdir, f'run_{args.mode}_{ts}')
    os.makedirs(run_out, exist_ok=True)

    if args.mode == 'global':
        logger.info('Running in GLOBAL ECC mode')
        res = process_global_mode(frames, mask_img, run_out, motion_model=motion_model, pyr_levels=args.pyr, logger=logger)
        logger.info(f'Global mode finished. Outputs in {run_out}')
    else:
        logger.info('Running in GRID ECC mode')
        grid_rows, grid_cols = args.grid
        res = process_grid_mode(frames, mask_img, run_out, grid_rows=grid_rows, grid_cols=grid_cols, motion_model=motion_model, logger=logger)
        logger.info(f'Grid mode finished. Outputs in {run_out}')

    logger.info('All done.')


if __name__ == '__main__':
    main()
