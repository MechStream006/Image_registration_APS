"""
test_registration.py
---------------------------------
Unit tests for registration functions in GUI_7.py

Usage:
    pytest -v test_registration.py
"""

import numpy as np
import cv2
import pytest
from GUI_7 import (
    phase_correlation_registration,
    ecc_registration,
    apply_translation_warp,
    apply_affine_warp,
    GPU_AVAILABLE,
)

# ========== Synthetic Image Utilities ==========

def make_test_image(size=256, seed=42):
    np.random.seed(seed)
    img = np.zeros((size, size), np.float32)
    cv2.circle(img, (size//2, size//2), size//4, 0.8, -1)
    cv2.line(img, (50, 50), (200, 150), 0.7, 3)
    cv2.rectangle(img, (30, 150), (90, 220), 0.5, -1)
    img += 0.05 * np.random.randn(size, size).astype(np.float32)
    return np.clip(img, 0, 1)

def apply_known_transform(img, dx, dy, angle=0.0, scale=1.0):
    """Apply a known affine transform to generate a 'moved' frame."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    moved = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return moved, M


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# ========== Tests ==========

def test_phase_correlation_translation_accuracy():
    ref = make_test_image()
    moved, _ = apply_known_transform(ref, dx=5.5, dy=-3.2)
    dy, dx = phase_correlation_registration(ref, moved)
    assert abs(dx - 5.5) < 0.5 and abs(dy - (-3.2)) < 0.5, f"Estimated: dx={dx}, dy={dy}"


def test_ecc_registration_affine_accuracy():
    ref = make_test_image()
    moved, true_M = apply_known_transform(ref, dx=3, dy=4, angle=5, scale=1.0)
    M = ecc_registration(ref, moved)
    registered = apply_affine_warp(moved, M)
    error = mse(ref, registered)
    assert error < 0.04, f"MSE too high: {error}"


def test_hybrid_registration_combination():
    """Hybrid: PCC estimate → ECC refine"""
    ref = make_test_image()
    moved, _ = apply_known_transform(ref, dx=7, dy=-6, angle=2)
    
    # Step 1: PCC
    dy, dx = phase_correlation_registration(ref, moved)
    warp_matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    
    # Step 2: ECC refinement
    try:
        _, warp_matrix = cv2.findTransformECC(
            (ref * 255).astype(np.uint8),
            (moved * 255).astype(np.uint8),
            warp_matrix,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        )
    except cv2.error:
        pytest.skip("ECC failed to converge.")
    
    registered = apply_affine_warp(moved, warp_matrix)
    err = mse(ref, registered)
    assert err < 0.06, f"Hybrid registration MSE too high: {err}"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available for comparison test")
def test_gpu_vs_cpu_consistency():
    """Ensure GPU and CPU phase correlation give similar results"""
    ref = make_test_image()
    moved, _ = apply_known_transform(ref, dx=4, dy=-2)
    
    # Force CPU
    import GUI_7 as g7
    g7.GPU_AVAILABLE = False
    dy_cpu, dx_cpu = g7.phase_correlation_registration(ref, moved)
    
    # Force GPU
    g7.GPU_AVAILABLE = True
    dy_gpu, dx_gpu = g7.phase_correlation_registration(ref, moved)
    
    assert abs(dx_cpu - dx_gpu) < 0.2 and abs(dy_cpu - dy_gpu) < 0.2


def test_noise_robustness():
    """Ensure PCC still works with moderate Gaussian noise."""
    ref = make_test_image()
    moved, _ = apply_known_transform(ref, dx=3, dy=2)
    noisy = np.clip(moved + 0.05 * np.random.randn(*moved.shape), 0, 1)
    dy, dx = phase_correlation_registration(ref, noisy)
    assert abs(dx - 3) < 1 and abs(dy - 2) < 1, f"dx={dx}, dy={dy}"


def test_fail_gracefully_on_blank_input():
    """Registration should not crash on empty or constant images."""
    ref = np.zeros((128, 128), np.float32)
    moved = np.zeros_like(ref)
    dy, dx = phase_correlation_registration(ref, moved)
    assert np.isfinite(dy) and np.isfinite(dx)

