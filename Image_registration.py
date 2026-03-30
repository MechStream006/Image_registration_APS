import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import map_coordinates
import time
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ----------------- Image Preprocessing -----------------
# =========================================================

def preprocess_for_registration(img):
    """Normalize and enhance image for registration"""
    # Convert to float32
    img_f = img.astype(np.float32)
    
    # Robust normalization (clip outliers)
    p2, p98 = np.percentile(img_f, (2, 98))
    img_clip = np.clip(img_f, p2, p98)
    
    # Normalize to [0, 1]
    img_norm = (img_clip - p2) / (p98 - p2 + 1e-8)
    
    return img_norm

def create_pyramid(img, levels=3):
    """Create multi-resolution pyramid"""
    pyramid = [img]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.insert(0, img)
    return pyramid

# =========================================================
# ----------------- Registration Algorithms -----------------
# =========================================================

class RegistrationAlgorithm:
    """Base class for registration algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.transform_matrix = None
        self.execution_time = 0
        self.pyramid_levels = 3
        
    def register(self, mask_frame: np.ndarray, live_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns: (registered_frame, affine_matrix)"""
        raise NotImplementedError


class EnhancedNCC(RegistrationAlgorithm):
    """Enhanced NCC with Multi-Resolution Pyramid"""
    
    def __init__(self, max_iter=100):
        super().__init__("Enhanced NCC (Multi-Resolution)")
        self.max_iter = max_iter
    
    def ncc_metric(self, params, mask, live, h, w):
        """Calculate NCC similarity"""
        try:
            transformed = self.apply_transform(live, params, h, w)
            
            # Normalized Cross-Correlation
            mask_mean = np.mean(mask)
            live_mean = np.mean(transformed)
            
            numerator = np.sum((mask - mask_mean) * (transformed - live_mean))
            denominator = np.sqrt(np.sum((mask - mask_mean)**2) * np.sum((transformed - live_mean)**2))
            
            if denominator < 1e-10:
                return 1e10
            
            ncc = numerator / denominator
            return -ncc  # Negative for minimization
            
        except:
            return 1e10
    
    def apply_transform(self, img, params, h, w):
        """Apply similarity transform (4 DOF: tx, ty, rotation, scale)"""
        tx, ty, angle, scale = params
        
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        transformed = cv2.warpAffine(img, M, (w, h), 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        return transformed
    
    def register(self, mask_frame, live_frame):
        start_time = time.time()
        
        # Preprocess
        mask_norm = preprocess_for_registration(mask_frame)
        live_norm = preprocess_for_registration(live_frame)
        
        # Create pyramids
        mask_pyramid = create_pyramid(mask_norm, self.pyramid_levels)
        live_pyramid = create_pyramid(live_norm, self.pyramid_levels)
        
        # Initialize parameters [tx, ty, angle, scale]
        params = [0.0, 0.0, 0.0, 1.0]
        
        # Coarse to fine registration
        for level in range(len(mask_pyramid)):
            mask_level = mask_pyramid[level]
            live_level = live_pyramid[level]
            h, w = mask_level.shape
            
            # Scale factor for this pyramid level
            scale_factor = 2 ** level
            
            # Scale translation parameters
            if level > 0:
                params[0] *= 2
                params[1] *= 2
            
            # Optimization bounds
            max_translation = min(h, w) * 0.2 / scale_factor
            bounds = [
                (-max_translation, max_translation),  # tx
                (-max_translation, max_translation),  # ty
                (-np.pi/6, np.pi/6),                  # angle (±30 degrees)
                (0.8, 1.2)                            # scale
            ]
            
            # Optimize at this level
            result = minimize(
                self.ncc_metric,
                params,
                args=(mask_level, live_level, h, w),
                method='Powell',
                bounds=bounds,
                options={'maxiter': self.max_iter // (level + 1), 'disp': False}
            )
            
            params = result.x
        
        # Apply final transformation to original image
        h, w = mask_frame.shape
        registered = self.apply_transform(live_frame, params, h, w)
        
        # Build affine matrix
        tx, ty, angle, scale = params
        center = (w / 2, h / 2)
        M_2x3 = cv2.getRotationMatrix2D(center, np.degrees(angle), scale)
        M_2x3[0, 2] += tx
        M_2x3[1, 2] += ty
        
        affine_matrix = np.vstack([M_2x3, [0, 0, 1]])
        
        self.execution_time = time.time() - start_time
        self.transform_matrix = affine_matrix
        
        return registered, affine_matrix


class EnhancedMI(RegistrationAlgorithm):
    """Enhanced Mutual Information with Multi-Resolution"""
    
    def __init__(self, bins=32, max_iter=100):
        super().__init__("Enhanced MI (Multi-Resolution)")
        self.bins = bins
        self.max_iter = max_iter
    
    def mutual_information(self, img1, img2):
        """Calculate Mutual Information efficiently"""
        try:
            # Create joint histogram
            hist_2d, _, _ = np.histogram2d(
                img1.ravel(), 
                img2.ravel(), 
                bins=self.bins,
                range=[[0, 1], [0, 1]]
            )
            
            # Smooth histogram
            hist_2d = hist_2d + 1e-10
            
            # Normalize
            pxy = hist_2d / np.sum(hist_2d)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            
            # Calculate entropies
            px_py = px[:, None] * py[None, :]
            
            # MI calculation
            nzs = pxy > 1e-10
            mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
            
            return mi
        except:
            return -1e10
    
    def mi_metric(self, params, mask, live, h, w):
        """MI metric for optimization"""
        try:
            transformed = self.apply_transform(live, params, h, w)
            mi = self.mutual_information(mask, transformed)
            return -mi  # Negative for minimization
        except:
            return 1e10
    
    def apply_transform(self, img, params, h, w):
        """Apply similarity transform"""
        tx, ty, angle, scale = params
        
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        transformed = cv2.warpAffine(img, M, (w, h), 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        return transformed
    
    def register(self, mask_frame, live_frame):
        start_time = time.time()
        
        # Preprocess
        mask_norm = preprocess_for_registration(mask_frame)
        live_norm = preprocess_for_registration(live_frame)
        
        # Create pyramids
        mask_pyramid = create_pyramid(mask_norm, self.pyramid_levels)
        live_pyramid = create_pyramid(live_norm, self.pyramid_levels)
        
        # Initialize
        params = [0.0, 0.0, 0.0, 1.0]
        
        # Multi-resolution optimization
        for level in range(len(mask_pyramid)):
            mask_level = mask_pyramid[level]
            live_level = live_pyramid[level]
            h, w = mask_level.shape
            
            scale_factor = 2 ** level
            
            if level > 0:
                params[0] *= 2
                params[1] *= 2
            
            max_translation = min(h, w) * 0.2 / scale_factor
            bounds = [
                (-max_translation, max_translation),
                (-max_translation, max_translation),
                (-np.pi/6, np.pi/6),
                (0.8, 1.2)
            ]
            
            result = minimize(
                self.mi_metric,
                params,
                args=(mask_level, live_level, h, w),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iter // (level + 1), 'disp': False}
            )
            
            params = result.x
        
        # Apply to original
        h, w = mask_frame.shape
        registered = self.apply_transform(live_frame, params, h, w)
        
        # Build matrix
        tx, ty, angle, scale = params
        center = (w / 2, h / 2)
        M_2x3 = cv2.getRotationMatrix2D(center, np.degrees(angle), scale)
        M_2x3[0, 2] += tx
        M_2x3[1, 2] += ty
        
        affine_matrix = np.vstack([M_2x3, [0, 0, 1]])
        
        self.execution_time = time.time() - start_time
        self.transform_matrix = affine_matrix
        
        return registered, affine_matrix


class EnhancedFeatureBased(RegistrationAlgorithm):
    """Enhanced Feature-Based with Preprocessing"""
    
    def __init__(self, feature_type='ORB', min_matches=10):
        super().__init__(f"Enhanced Feature-Based ({feature_type})")
        self.feature_type = feature_type
        self.min_matches = min_matches
    
    def register(self, mask_frame, live_frame):
        start_time = time.time()
        
        # Preprocess and enhance
        mask_prep = preprocess_for_registration(mask_frame)
        live_prep = preprocess_for_registration(live_frame)
        
        # Convert to uint8 with proper scaling
        mask_u8 = (mask_prep * 255).astype(np.uint8)
        live_u8 = (live_prep * 255).astype(np.uint8)
        
        # Apply CLAHE for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        mask_u8 = clahe.apply(mask_u8)
        live_u8 = clahe.apply(live_u8)
        
        # Feature detection
        if self.feature_type == 'ORB':
            detector = cv2.ORB_create(nfeatures=2000, 
                                     scaleFactor=1.2,
                                     nlevels=8,
                                     edgeThreshold=15)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            detector = cv2.SIFT_create(nfeatures=2000)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        kp1, des1 = detector.detectAndCompute(mask_u8, None)
        kp2, des2 = detector.detectAndCompute(live_u8, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Fallback: return identity transform
            print(f"  ⚠️  {self.name}: Insufficient features, returning identity")
            self.execution_time = time.time() - start_time
            self.transform_matrix = np.eye(3)
            return live_frame, np.eye(3)
        
        # Match features with ratio test
        matches = matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            print(f"  ⚠️  {self.name}: Only {len(good_matches)} matches, returning identity")
            self.execution_time = time.time() - start_time
            self.transform_matrix = np.eye(3)
            return live_frame, np.eye(3)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate transformation with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, 
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.99
        )
        
        if M is None or inliers is None:
            print(f"  ⚠️  {self.name}: RANSAC failed, returning identity")
            self.execution_time = time.time() - start_time
            self.transform_matrix = np.eye(3)
            return live_frame, np.eye(3)
        
        inlier_count = np.sum(inliers)
        if inlier_count < self.min_matches:
            print(f"  ⚠️  {self.name}: Only {inlier_count} inliers, returning identity")
            self.execution_time = time.time() - start_time
            self.transform_matrix = np.eye(3)
            return live_frame, np.eye(3)
        
        # Apply transformation
        h, w = mask_frame.shape
        registered = cv2.warpAffine(live_frame, M, (w, h), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        
        affine_matrix = np.vstack([M, [0, 0, 1]])
        
        self.execution_time = time.time() - start_time
        self.transform_matrix = affine_matrix
        
        print(f"  ✓ {self.name}: {inlier_count}/{len(good_matches)} inliers")
        
        return registered, affine_matrix


class EnhancedPhaseCorrelation(RegistrationAlgorithm):
    """Enhanced Phase Correlation with Sub-pixel Accuracy"""
    
    def __init__(self):
        super().__init__("Enhanced Phase Correlation")
    
    def register(self, mask_frame, live_frame):
        start_time = time.time()
        
        # Preprocess
        mask_norm = preprocess_for_registration(mask_frame)
        live_norm = preprocess_for_registration(live_frame)
        
        # Use pyramid for better accuracy
        mask_pyr = create_pyramid(mask_norm, 3)
        live_pyr = create_pyramid(live_norm, 3)
        
        # Start from coarsest level
        total_shift_x = 0.0
        total_shift_y = 0.0
        
        for level in range(len(mask_pyr)):
            mask_level = mask_pyr[level]
            live_level = live_pyr[level]
            
            # Apply current shift estimate
            if level > 0:
                total_shift_x *= 2
                total_shift_y *= 2
                M = np.float32([[1, 0, total_shift_x], [0, 1, total_shift_y]])
                h, w = mask_level.shape
                live_level = cv2.warpAffine(live_level, M, (w, h))
            
            # Phase correlation
            shift, response = cv2.phaseCorrelate(
                mask_level.astype(np.float32),
                live_level.astype(np.float32)
            )
            
            total_shift_x += shift[0]
            total_shift_y += shift[1]
        
        # Apply final shift
        h, w = mask_frame.shape
        M_2x3 = np.float32([[1, 0, total_shift_x], [0, 1, total_shift_y]])
        registered = cv2.warpAffine(live_frame, M_2x3, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        
        affine_matrix = np.vstack([M_2x3, [0, 0, 1]])
        
        self.execution_time = time.time() - start_time
        self.transform_matrix = affine_matrix
        
        return registered, affine_matrix


# =========================================================
# ----------------- Metrics & Comparison -----------------
# =========================================================

def calculate_registration_metrics(mask, registered, original_live):
    """Calculate comprehensive quality metrics"""
    
    mask_f = mask.astype(np.float32)
    reg_f = registered.astype(np.float32)
    
    # DSA Image
    dsa_image = mask_f - reg_f
    
    # Mean Squared Error
    mse = np.mean((mask_f - reg_f) ** 2)
    
    # Normalized Cross-Correlation
    mask_norm = (mask_f - np.mean(mask_f)) / (np.std(mask_f) + 1e-8)
    reg_norm = (reg_f - np.mean(reg_f)) / (np.std(reg_f) + 1e-8)
    ncc = np.sum(mask_norm * reg_norm) / mask_norm.size
    
    # Mutual Information
    mask_01 = preprocess_for_registration(mask_f)
    reg_01 = preprocess_for_registration(reg_f)
    hist_2d, _, _ = np.histogram2d(mask_01.ravel(), reg_01.ravel(), bins=32)
    pxy = hist_2d / (np.sum(hist_2d) + 1e-10)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 1e-10
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px_py[nzs] + 1e-10)))
    
    # DSA Quality
    dsa_mean_abs = np.mean(np.abs(dsa_image))
    dsa_std = np.std(dsa_image)
    
    # Structural Similarity (simplified SSIM)
    c1, c2 = (0.01 * np.max(mask_f))**2, (0.03 * np.max(mask_f))**2
    mu1, mu2 = np.mean(mask_f), np.mean(reg_f)
    sigma1_sq = np.var(mask_f)
    sigma2_sq = np.var(reg_f)
    sigma12 = np.mean((mask_f - mu1) * (reg_f - mu2))
    ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return {
        'MSE': float(mse),
        'NCC': float(ncc),
        'MI': float(mi),
        'SSIM': float(ssim),
        'DSA_Mean_Abs': float(dsa_mean_abs),
        'DSA_Std': float(dsa_std),
        'DSA_Image': dsa_image
    }


def compare_registration_algorithms(mask_frame, live_frame, output_dir='registration_results'):
    """Compare all registration algorithms"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize algorithms
    algorithms = [
        EnhancedNCC(max_iter=100),
        EnhancedMI(bins=32, max_iter=100),
        EnhancedFeatureBased(feature_type='ORB'),
        EnhancedPhaseCorrelation()
    ]
    
    results = {}
    
    print("=" * 80)
    print("ENHANCED REGISTRATION ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Image size: {mask_frame.shape}")
    print(f"Value range: Mask [{np.min(mask_frame):.1f}, {np.max(mask_frame):.1f}], "
          f"Live [{np.min(live_frame):.1f}, {np.max(live_frame):.1f}]")
    print("=" * 80)
    
    # Process each algorithm
    for algo in algorithms:
        print(f"\n🔄 Processing: {algo.name}")
        
        try:
            registered, affine_matrix = algo.register(mask_frame, live_frame)
            metrics = calculate_registration_metrics(mask_frame, registered, live_frame)
            
            results[algo.name] = {
                'algorithm': algo,
                'registered_frame': registered,
                'affine_matrix': affine_matrix,
                'metrics': metrics,
                'execution_time': algo.execution_time
            }
            
            print(f"  ✅ Time: {algo.execution_time:.3f}s")
            print(f"  📊 MSE: {metrics['MSE']:.2f} | NCC: {metrics['NCC']:.4f} | MI: {metrics['MI']:.4f} | SSIM: {metrics['SSIM']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\n❌ All algorithms failed!")
        return None
    
    # Generate outputs
    visualize_comparison(mask_frame, live_frame, results, output_dir)
    generate_metrics_report(results, output_dir)
    
    print(f"\n✅ Results saved to: {output_dir}")
    return results


def visualize_comparison(mask_frame, live_frame, results, output_dir):
    """Create visual comparisons"""
    
    n_algos = len(results)
    
    # Figure 1: Registration Results + Difference
    fig1, axes = plt.subplots(3, n_algos + 1, figsize=(4*(n_algos+1), 12))
    
    # Column 0: Original
    axes[0, 0].imshow(mask_frame, cmap='gray')
    axes[0, 0].set_title('Mask Frame\n(Pre-Contrast)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(live_frame, cmap='gray')
    axes[1, 0].set_title('Live Frame\n(Unregistered)', fontweight='bold')
    axes[1, 0].axis('off')
    
    diff_unreg = np.abs(mask_frame - live_frame)
    axes[2, 0].imshow(diff_unreg, cmap='hot')
    axes[2, 0].set_title('Difference\n(No Registration)', fontweight='bold')
    axes[2, 0].axis('off')
    
    # Registered results
    for i, (name, data) in enumerate(results.items(), 1):
        # Registered frame
        axes[0, i].imshow(data['registered_frame'], cmap='gray')
        axes[0, i].set_title(f"{name}\n(Registered)", fontsize=9, fontweight='bold')
        axes[0, i].axis('off')
        
        # Overlay
        overlay = cv2.addWeighted(
            cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            0.5,
            cv2.normalize(data['registered_frame'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            0.5, 0
        )
        axes[1, i].imshow(overlay, cmap='gray')
        axes[1, i].set_title(f"Overlay\nNCC: {data['metrics']['NCC']:.4f}", fontsize=9)
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(mask_frame - data['registered_frame'])
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f"Difference\nMSE: {data['metrics']['MSE']:.2f}", fontsize=9)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_registration_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  💾 Saved: 01_registration_comparison.png")
    
    # Figure 2: DSA Results
    fig2, axes = plt.subplots(1, n_algos + 1, figsize=(5*(n_algos+1), 5))
    
    dsa_unreg = mask_frame.astype(np.float32) - live_frame.astype(np.float32)
    vmin, vmax = np.percentile(dsa_unreg, [1, 99])
    
    axes[0].imshow(dsa_unreg, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('DSA\n(No Registration)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    for i, (name, data) in enumerate(results.items(), 1):
        dsa_img = data['metrics']['DSA_Image']
        axes[i].imshow(dsa_img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{name}\nDSA Result", fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_dsa_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  💾 Saved: 02_dsa_comparison.png")
    
    # Figure 3: Transformation Matrices
    fig3, axes = plt.subplots(1, n_algos, figsize=(5*n_algos, 5))
    if n_algos == 1:
        axes = [axes]
    
    for i, (name, data) in enumerate(results.items()):
        matrix = data['affine_matrix']
        im = axes[i].imshow(matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        axes[i].set_title(f"{name}\nAffine Matrix", fontsize=11, fontweight='bold')
        
        for row in range(3):
            for col in range(3):
                color = 'white' if abs(matrix[row, col]) > 1 else 'black'
                axes[i].text(col, row, f'{matrix[row, col]:.4f}',
                           ha='center', va='center', fontsize=10,
                           color=color, fontweight='bold')
        
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_yticks([0, 1, 2])
        axes[i].set_xticklabels(['X', 'Y', 'T'])
        axes[i].set_yticklabels(['X', 'Y', '1'])
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_affine_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  💾 Saved: 03_affine_matrices.png")


def generate_metrics_report(results, output_dir):
    """Generate Excel report"""
    
    metrics_data = []
    for name, data in results.items():
        row = {
            'Algorithm': name,
            'Execution_Time_s': data['execution_time'],
            'MSE': data['metrics']['MSE'],
            'NCC': data['metrics']['NCC'],
            'MI': data['metrics']['MI'],
            'SSIM': data['metrics']['SSIM'],
            'DSA_Mean_Abs': data['metrics']['DSA_Mean_Abs'],
            'DSA_Std': data['metrics']['DSA_Std']
        }
        metrics_data.append(row)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Matrix details
    matrix_data = []
    for name, data in results.items():
        matrix = data['affine_matrix']
        matrix_data.append({
            'Algorithm': name,
            'M00': matrix[0, 0],
            'M01': matrix[0, 1],
            'M02_TX': matrix[0, 2],
            'M10': matrix[1, 0],
            'M11': matrix[1, 1],
            'M12_TY': matrix[1, 2]
        })
    
    df_matrices = pd.DataFrame(matrix_data)
    
    # Save to Excel
    excel_path = f"{output_dir}/registration_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics_Comparison', index=False)
        df_matrices.to_excel(writer, sheet_name='Affine_Matrices', index=False)
        
        # Ranking
        df_ranked = df_metrics.copy()
        df_ranked['MSE_Rank'] = df_ranked['MSE'].rank()
        df_ranked['NCC_Rank'] = df_ranked['NCC'].rank(ascending=False)
        df_ranked['MI_Rank'] = df_ranked['MI'].rank(ascending=False)
        df_ranked['SSIM_Rank'] = df_ranked['SSIM'].rank(ascending=False)
        df_ranked['Speed_Rank'] = df_ranked['Execution_Time_s'].rank()
        df_ranked['Overall_Score'] = (
            df_ranked['MSE_Rank'] + 
            df_ranked['NCC_Rank'] + 
            df_ranked['MI_Rank'] + 
            df_ranked['SSIM_Rank'] + 
            df_ranked['Speed_Rank']
        ) / 5
        df_ranked = df_ranked.sort_values('Overall_Score')
        
        df_ranked.to_excel(writer, sheet_name='Ranking', index=False)
    
    print(f"  💾 Saved: registration_metrics.xlsx")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RANKING SUMMARY (Lower Overall_Score is Better)")
    print("=" * 80)
    print(df_ranked[['Algorithm', 'Overall_Score', 'MSE', 'NCC', 'MI', 'SSIM', 'Execution_Time_s']].to_string(index=False))
    print("=" * 80)


# =========================================================
# ----------------- Main Processing Function -----------------
# =========================================================

def process_dsa_with_registration(dicom_path, frame_idx_mask=0, frame_idx_live=10, 
                                  output_dir='dsa_registration_output'):
    """Complete DSA processing with registration comparison"""
    
    print("=" * 80)
    print("DSA IMAGE REGISTRATION PIPELINE")
    print("=" * 80)
    print(f"📁 DICOM: {dicom_path}")
    print(f"🎯 Mask Frame: {frame_idx_mask}, Live Frame: {frame_idx_live}")
    print("=" * 80)
    
    # Load DICOM
    try:
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array
        
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        
        print(f"✅ Loaded: {arr.shape[0]} frames, {arr.shape[1]}x{arr.shape[2]} pixels")
        print(f"   Pixel type: {arr.dtype}, Range: [{np.min(arr)}, {np.max(arr)}]")
        
        # Validate frame indices
        if frame_idx_mask >= arr.shape[0] or frame_idx_live >= arr.shape[0]:
            print(f"❌ Error: Frame indices out of range (max: {arr.shape[0]-1})")
            return None
        
        # Extract frames
        mask_frame = arr[frame_idx_mask].astype(np.float32)
        live_frame = arr[frame_idx_live].astype(np.float32)
        
        print(f"✅ Extracted frames successfully")
        
    except Exception as e:
        print(f"❌ Error loading DICOM: {e}")
        return None
    
    # Run registration comparison
    results = compare_registration_algorithms(mask_frame, live_frame, output_dir)
    
    return results


# =========================================================
# ----------------- Usage Example -----------------
# =========================================================

if __name__ == "__main__":
    
    # Your DICOM path
    dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
    

    results = process_dsa_with_registration(
    dicom_path=dicom_path,
    output_dir="D:/Rohith/Auto_pixel_shift/APS/synthetic_test_output/DSA_Registration"
    )
    if os.path.exists(dicom_path):
        print("\n🏥 Processing Real DICOM File...\n")
        results = process_dsa_with_registration(
            dicom_path=dicom_path,
            frame_idx_mask=0,
            frame_idx_live=10,
            output_dir="registration_comparison_output"
        )
        
        if results:
            print("\n" + "=" * 80)
            print("✅ PROCESSING COMPLETE!")
            print("=" * 80)
            print("📂 Output files:")
            print("   - 01_registration_comparison.png (Registered frames + differences)")
            print("   - 02_dsa_comparison.png (DSA results)")
            print("   - 03_affine_matrices.png (Transformation matrices)")
            print("   - registration_metrics.xlsx (Quantitative metrics)")
            print("=" * 80)
    else:
        print(f"⚠️  DICOM file not found: {dicom_path}")
        print("\n🧪 Running Synthetic Test...\n")
        
        # Create synthetic test with known transformation
        np.random.seed(42)
        size = 512
        
        # Create synthetic angiography image
        mask = np.zeros((size, size), dtype=np.float32)
        
        # Add vessel-like structures
        for _ in range(20):
            x1, y1 = np.random.randint(50, size-50, 2)
            x2, y2 = x1 + np.random.randint(-100, 100), y1 + np.random.randint(-100, 100)
            cv2.line(mask, (x1, y1), (x2, y2), 
                    color=np.random.randint(500, 1500), 
                    thickness=np.random.randint(2, 8))
        
        # Add circles (vessel cross-sections)
        for _ in range(15):
            x, y = np.random.randint(50, size-50, 2)
            cv2.circle(mask, (x, y), 
                      np.random.randint(5, 20), 
                      np.random.randint(600, 1800), -1)
        
        # Smooth
        mask = cv2.GaussianBlur(mask, (15, 15), 3)
        mask = mask + np.random.randn(size, size).astype(np.float32) * 50
        
        # Create live frame with known transformation
        # Apply: translation (10, -5), rotation (3 degrees), scale (1.02)
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, 3.0, 1.02)
        M[0, 2] += 10  # tx
        M[1, 2] += -5  # ty
        
        live = cv2.warpAffine(mask, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        live = live + np.random.randn(size, size).astype(np.float32) * 80
        
        print("✅ Created synthetic DSA images with known transformation:")
        print(f"   Translation: (10, -5), Rotation: 3°, Scale: 1.02")
        print(f"   Added noise for realism")
        print()
        
        results = compare_registration_algorithms(mask, live, 'synthetic_test_output')
        
        if results:
            print("\n✅ Synthetic test completed successfully!")
            print("   Check 'synthetic_test_output' folder for results")