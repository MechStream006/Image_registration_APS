import numpy as np
import cv2
import pydicom
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
from scipy import ndimage
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnatomyConfig:
    """Configuration parameters for different anatomy types"""
    name: str
    orb_features: int
    orb_scale_factor: float
    orb_n_levels: int
    ransac_threshold: float
    ransac_max_iter: int
    match_ratio_threshold: float
    roi_margin: int  
    preprocessing_blur: float
    clahe_clip_limit: float
    
    @staticmethod
    def get_config(anatomy_type: str) -> 'AnatomyConfig':
        """Get anatomy-specific configuration"""
        configs = {
            'neuro': AnatomyConfig(
                name='Neuro (Head/Brain)',
                orb_features=3000,
                orb_scale_factor=1.2,
                orb_n_levels=8,
                ransac_threshold=2.0,
                ransac_max_iter=2000,
                match_ratio_threshold=0.75,
                roi_margin=50,
                preprocessing_blur=1.0,
                clahe_clip_limit=3.0
            ),
            'peripheral': AnatomyConfig(
                name='Peripheral (Limbs)',
                orb_features=2500,
                orb_scale_factor=1.2,
                orb_n_levels=8,
                ransac_threshold=2.5,
                ransac_max_iter=1500,
                match_ratio_threshold=0.75,
                roi_margin=40,
                preprocessing_blur=1.2,
                clahe_clip_limit=2.5
            ),
            'gi': AnatomyConfig(
                name='GI (Gastrointestinal)',
                orb_features=2000,
                orb_scale_factor=1.15,
                orb_n_levels=8,
                ransac_threshold=3.0,
                ransac_max_iter=1500,
                match_ratio_threshold=0.70,
                roi_margin=60,
                preprocessing_blur=1.5,
                clahe_clip_limit=2.0
            ),
            'cardiac': AnatomyConfig(
                name='Cardiac (Heart)',
                orb_features=2500,
                orb_scale_factor=1.2,
                orb_n_levels=8,
                ransac_threshold=2.0,
                ransac_max_iter=2000,
                match_ratio_threshold=0.75,
                roi_margin=50,
                preprocessing_blur=1.0,
                clahe_clip_limit=2.5
            )
        }
        
        if anatomy_type.lower() not in configs:
            logger.warning(f"Unknown anatomy type '{anatomy_type}', using 'peripheral'")
            return configs['peripheral']
        
        return configs[anatomy_type.lower()]


class DICOMLoader:
    """Handle DICOM sequence loading and metadata extraction"""
    
    @staticmethod
    def load_sequence(input_path: Path) -> Tuple[List[np.ndarray], List[pydicom.Dataset]]:
        """
        Load DICOM sequence from directory or single file
        
        Returns:
            frames: List of pixel arrays (uint16)
            datasets: List of DICOM dataset objects
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            # Single multi-frame DICOM
            logger.info(f"Loading multi-frame DICOM: {input_path}")
            ds = pydicom.dcmread(str(input_path))
            
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                frames = []
                for i in range(ds.NumberOfFrames):
                    frame = ds.pixel_array[i]
                    frames.append(frame)
                return frames, [ds] * len(frames)
            else:
                return [ds.pixel_array], [ds]
        
        elif input_path.is_dir():
            # Directory of DICOM files
            logger.info(f"Loading DICOM sequence from directory: {input_path}")
            dicom_files = sorted(input_path.glob("*.dcm"))
            
            if not dicom_files:
                dicom_files = sorted(input_path.glob("*"))
                dicom_files = [f for f in dicom_files if f.is_file()]
            
            frames = []
            datasets = []
            
            for dicom_file in dicom_files:
                try:
                    ds = pydicom.dcmread(str(dicom_file))
                    frames.append(ds.pixel_array)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to read {dicom_file}: {e}")
            
            logger.info(f"Loaded {len(frames)} frames")
            return frames, datasets
        
        else:
            raise ValueError(f"Invalid input path: {input_path}")
    
    @staticmethod
    def save_sequence(frames: List[np.ndarray], 
                 original_datasets: List[pydicom.Dataset],
                 output_path: Path,
                 description: str = "DSA Corrected"):
        """
    Save processed frames as DICOM sequence
    
    Args:
        frames: Processed pixel arrays
        original_datasets: Original DICOM datasets for metadata
        output_path: Output directory
        description: Series description
    """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
        logger.info(f"Saving {len(frames)} frames to {output_path}")
    
        for i, (frame, original_ds) in enumerate(zip(frames, original_datasets)):
        # Create new dataset based on original
            ds = original_ds.copy()
        
        # Update pixel data
            ds.PixelData = frame.tobytes()
            ds.Rows, ds.Columns = frame.shape
        
        # Update metadata
            ds.SeriesDescription = description
            ds.SeriesNumber = original_ds.SeriesNumber + 1000 if hasattr(original_ds, 'SeriesNumber') else 9999
            ds.InstanceNumber = i + 1
        
        # Save using modern flag (avoid deprecation warning)
            ds.save_as(str(output_file := output_path / f"frame_{i:04d}.dcm"), enforce_file_format=True)
    
        logger.info(f"DICOM sequence saved successfully")



class DSAPreprocessor:
    """Phase 1: Preprocessing pipeline"""
    
    def __init__(self, config: AnatomyConfig):
        self.config = config
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame
        
        Steps:
        1. Normalize to 8-bit
        2. Apply CLAHE for contrast enhancement
        3. Denoise with bilateral filter
        4. Apply Gaussian smoothing
        """
        # Convert to 8-bit
        frame_normalized = self._normalize_to_8bit(frame)
        
        # CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(8, 8)
        )
        frame_clahe = clahe.apply(frame_normalized)
        
        # Bilateral filter for edge-preserving denoising
        frame_denoised = cv2.bilateralFilter(
            frame_clahe,
            d=5,
            sigmaColor=50,
            sigmaSpace=50
        )
        
        # Gentle Gaussian smoothing
        if self.config.preprocessing_blur > 0:
            frame_smooth = gaussian(
                frame_denoised,
                sigma=self.config.preprocessing_blur,
                preserve_range=True
            ).astype(np.uint8)
        else:
            frame_smooth = frame_denoised
        
        return frame_smooth
    
    @staticmethod
    def _normalize_to_8bit(frame: np.ndarray) -> np.ndarray:
        """Normalize frame to 8-bit range with robust scaling"""
        # Use percentile-based normalization to handle outliers
        p_low = np.percentile(frame, 1)
        p_high = np.percentile(frame, 99)
        
        frame_clipped = np.clip(frame, p_low, p_high)
        
        if p_high > p_low:
            frame_normalized = ((frame_clipped - p_low) / (p_high - p_low) * 255)
        else:
            frame_normalized = np.zeros_like(frame_clipped)
        
        return frame_normalized.astype(np.uint8)
    
    def create_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create ROI mask excluding borders and collimators
        
        Returns:
            Binary mask (255 = valid region, 0 = exclude)
        """
        h, w = frame.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Exclude borders
        margin = self.config.roi_margin
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
        
        # Detect collimator regions (very dark areas)
        threshold = np.percentile(frame, 5)
        collimator_mask = frame < threshold
        mask[collimator_mask] = 0
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask


class ORBFeatureDetector:
    """Phase 2: ORB feature detection and matching"""
    
    def __init__(self, config: AnatomyConfig):
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.orb_features,
            scaleFactor=config.orb_scale_factor,
            nlevels=config.orb_n_levels,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Brute force matcher with Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_compute(self, 
                          frame: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Detect ORB features and compute descriptors
        
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: (N, 32) array of binary descriptors
        """
        keypoints, descriptors = self.orb.detectAndCompute(frame, mask)
        
        if descriptors is None:
            logger.warning("No features detected!")
            return [], None
        
        logger.debug(f"Detected {len(keypoints)} features")
        return keypoints, descriptors
    
    def match_features(self,
                      desc1: np.ndarray,
                      desc2: np.ndarray) -> List[Tuple[int, int]]:
        """
        Match features using Lowe's ratio test
        
        Returns:
            List of (query_idx, train_idx) matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Get 2 nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))
        
        logger.debug(f"Good matches after ratio test: {len(good_matches)}")
        return good_matches


class RANSACEstimator:
    """Phase 3: RANSAC-based robust transformation estimation"""
    
    def __init__(self, config: AnatomyConfig):
        self.config = config
    
    def estimate_affine_transform(self,
                                 keypoints_src: List,
                                 keypoints_dst: List,
                                 matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate affine transformation using RANSAC
        
        Returns:
            transform_matrix: (2, 3) affine transformation matrix
            inlier_mask: Boolean array indicating inliers
        """
        if len(matches) < 4:
            logger.warning(f"Insufficient matches ({len(matches)}) for RANSAC")
            return np.eye(2, 3, dtype=np.float32), np.array([])
        
        # Extract matched point coordinates
        src_pts = np.float32([keypoints_src[m[0]].pt for m in matches])
        dst_pts = np.float32([keypoints_dst[m[1]].pt for m in matches])
        
        # RANSAC to estimate affine transformation
        transform_matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
            maxIters=self.config.ransac_max_iter,
            confidence=0.99,
            refineIters=10
        )
        
        if transform_matrix is None:
            logger.warning("RANSAC failed, using identity transform")
            return np.eye(2, 3, dtype=np.float32), np.array([])
        
        inliers = np.sum(inlier_mask)
        inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0
        
        logger.info(f"RANSAC: {inliers}/{len(matches)} inliers ({inlier_ratio:.1%})")
        
        # Validate transformation (check for degenerate cases)
        if not self._is_valid_transform(transform_matrix):
            logger.warning("Invalid transformation detected, using identity")
            return np.eye(2, 3, dtype=np.float32), inlier_mask
        
        return transform_matrix, inlier_mask
    
    @staticmethod
    def _is_valid_transform(matrix: np.ndarray, 
                           max_scale: float = 1.5,
                           max_rotation: float = 30.0) -> bool:
        """
        Check if transformation is physically reasonable
        
        Args:
            max_scale: Maximum allowed scale change
            max_rotation: Maximum rotation in degrees
        """
        # Extract scale
        scale_x = np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
        scale_y = np.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2)
        
        if scale_x > max_scale or scale_y > max_scale:
            logger.warning(f"Scale too large: {scale_x:.2f}, {scale_y:.2f}")
            return False
        
        if scale_x < 1/max_scale or scale_y < 1/max_scale:
            logger.warning(f"Scale too small: {scale_x:.2f}, {scale_y:.2f}")
            return False
        
        # Extract rotation
        rotation_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
        rotation_deg = np.abs(np.degrees(rotation_rad))
        
        if rotation_deg > max_rotation:
            logger.warning(f"Rotation too large: {rotation_deg:.1f}°")
            return False
        
        return True


class MotionCompensator:
    """Phase 4: Apply transformations and warp images"""
    
    @staticmethod
    def warp_frame(frame: np.ndarray,
                  transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to frame
        
        Args:
            frame: Input image (uint8)
            transform_matrix: (2, 3) affine matrix
            
        Returns:
            Warped frame (uint8)
        """
        h, w = frame.shape
        
        warped = cv2.warpAffine(
            frame,
            transform_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped


class QualityEvaluator:
    """Phase 5: Evaluate registration quality"""
    
    @staticmethod
    def compute_mutual_information(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute normalized mutual information
        
        Higher values indicate better alignment
        """
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(
            img1.ravel(),
            img2.ravel(),
            bins=256,
            range=[[0, 256], [0, 256]]
        )
        
        # Convert to probability distribution
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Compute entropies
        px_py = px[:, None] * py[None, :]
        
        # Avoid log(0)
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        # Normalize
        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        
        nmi = 2 * mi / (hx + hy) if (hx + hy) > 0 else 0
        
        return nmi
    
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute structural similarity index"""
        return ssim(img1, img2, data_range=255)
    
    @staticmethod
    def compute_edge_sharpness(img: np.ndarray) -> float:
        """
        Compute edge sharpness using Laplacian variance
        
        Higher values indicate sharper edges (better DSA quality)
        """
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        return variance


class DSAGenerator:
    """Phase 6: Generate DSA images"""
    
    @staticmethod
    def subtract_images(contrast_frame: np.ndarray,
                       mask_frame: np.ndarray,
                       enhance: bool = True) -> np.ndarray:
        """
        Perform digital subtraction angiography
        
        Args:
            contrast_frame: Contrast-enhanced frame (uint8)
            mask_frame: Pre-contrast mask frame (uint8)
            enhance: Apply post-processing enhancement
            
        Returns:
            DSA image (uint8)
        """
        # Convert to float for subtraction
        contrast_float = contrast_frame.astype(np.float32)
        mask_float = mask_frame.astype(np.float32)
        
        # Subtraction
        dsa = contrast_float - mask_float
        
        # Clip negative values
        dsa = np.maximum(dsa, 0)
        
        if enhance:
            # Contrast stretching
            p_low = np.percentile(dsa[dsa > 0], 1) if np.any(dsa > 0) else 0
            p_high = np.percentile(dsa, 99)
            
            if p_high > p_low:
                dsa = np.clip((dsa - p_low) / (p_high - p_low) * 255, 0, 255)
            
            # Gentle smoothing to reduce noise
            dsa = gaussian(dsa, sigma=0.5, preserve_range=True)
        
        return dsa.astype(np.uint8)


class DSAProcessor:
    """Main pipeline orchestrator"""
    
    def __init__(self, anatomy_type: str = 'peripheral'):
        self.config = AnatomyConfig.get_config(anatomy_type)
        self.preprocessor = DSAPreprocessor(self.config)
        self.feature_detector = ORBFeatureDetector(self.config)
        self.ransac_estimator = RANSACEstimator(self.config)
        self.motion_compensator = MotionCompensator()
        self.quality_evaluator = QualityEvaluator()
        self.dsa_generator = DSAGenerator()
        
        logger.info(f"Initialized DSA Processor for: {self.config.name}")
    
    def process_sequence(self,
                        raw_frames: List[np.ndarray],
                        mask_frame_idx: int = 0) -> Dict:
        """
        Complete DSA motion correction pipeline
        
        Args:
            raw_frames: List of raw DICOM pixel arrays
            mask_frame_idx: Index of pre-contrast mask frame
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting DSA Processing Pipeline")
        logger.info(f"Anatomy: {self.config.name}")
        logger.info(f"Frames: {len(raw_frames)}")
        logger.info(f"Mask frame: {mask_frame_idx}")
        logger.info(f"{'='*70}\n")
        
        # Phase 1: Preprocessing
        logger.info("Phase 1: Preprocessing frames...")
        preprocessed_frames = []
        for i, frame in enumerate(raw_frames):
            processed = self.preprocessor.process_frame(frame)
            preprocessed_frames.append(processed)
            if i % 5 == 0:
                logger.info(f"  Preprocessed frame {i+1}/{len(raw_frames)}")
        
        # Create ROI mask
        roi_mask = self.preprocessor.create_roi_mask(preprocessed_frames[mask_frame_idx])
        
        # Phase 2: Feature detection on mask frame
        logger.info("\nPhase 2: Detecting features on mask frame...")
        mask_keypoints, mask_descriptors = self.feature_detector.detect_and_compute(
            preprocessed_frames[mask_frame_idx],
            roi_mask
        )
        
        if mask_descriptors is None:
            raise RuntimeError("Failed to detect features on mask frame!")
        
        logger.info(f"  Detected {len(mask_keypoints)} features on mask")
        
        # Phase 3 & 4: Register all frames to mask
        logger.info("\nPhase 3-4: Registering frames to mask...")
        registered_frames = []
        transformation_matrices = []
        registration_stats = []
        
        for i, frame in enumerate(preprocessed_frames):
            if i == mask_frame_idx:
                # Mask frame - no transformation needed
                registered_frames.append(frame.copy())
                transformation_matrices.append(np.eye(2, 3, dtype=np.float32))
                registration_stats.append({
                    'frame_idx': i,
                    'is_mask': True,
                    'n_matches': 0,
                    'n_inliers': 0
                })
                continue
            
            # Detect features in current frame
            frame_keypoints, frame_descriptors = self.feature_detector.detect_and_compute(
                frame, roi_mask
            )
            
            if frame_descriptors is None:
                logger.warning(f"  Frame {i}: No features detected, using identity transform")
                registered_frames.append(frame.copy())
                transformation_matrices.append(np.eye(2, 3, dtype=np.float32))
                registration_stats.append({
                    'frame_idx': i,
                    'is_mask': False,
                    'n_matches': 0,
                    'n_inliers': 0
                })
                continue
            
            # Match features
            matches = self.feature_detector.match_features(
                frame_descriptors,
                mask_descriptors
            )
            
            # Estimate transformation with RANSAC
            transform_matrix, inlier_mask = self.ransac_estimator.estimate_affine_transform(
                frame_keypoints,
                mask_keypoints,
                matches
            )
            
            # Warp frame
            registered = self.motion_compensator.warp_frame(frame, transform_matrix)
            
            registered_frames.append(registered)
            transformation_matrices.append(transform_matrix)
            
            n_inliers = np.sum(inlier_mask) if len(inlier_mask) > 0 else 0
            registration_stats.append({
                'frame_idx': i,
                'is_mask': False,
                'n_matches': len(matches),
                'n_inliers': n_inliers
            })
            
            logger.info(f"  Frame {i}: {len(matches)} matches, {n_inliers} inliers")
        
        # Phase 5: Quality evaluation
        logger.info("\nPhase 5: Evaluating registration quality...")
        quality_metrics = self._evaluate_quality(
            preprocessed_frames,
            registered_frames,
            mask_frame_idx
        )
        
        # Phase 6: Generate DSA images
        logger.info("\nPhase 6: Generating DSA images...")
        dsa_images = []
        mask_registered = registered_frames[mask_frame_idx]
        
        for i, frame in enumerate(registered_frames):
            if i == mask_frame_idx:
                # Mask frame - create empty DSA
                dsa_images.append(np.zeros_like(frame))
            else:
                dsa = self.dsa_generator.subtract_images(frame, mask_registered)
                dsa_images.append(dsa)
        
        logger.info(f"\n{'='*70}")
        logger.info("Processing Complete!")
        logger.info(f"{'='*70}\n")
        
        # Return comprehensive results
        return {
            'raw_frames': raw_frames,
            'preprocessed_frames': preprocessed_frames,
            'registered_frames': registered_frames,
            'dsa_images': dsa_images,
            'transformation_matrices': transformation_matrices,
            'registration_stats': registration_stats,
            'quality_metrics': quality_metrics,
            'mask_frame_idx': mask_frame_idx,
            'config': self.config,
            'roi_mask': roi_mask
        }
    
    def _evaluate_quality(self,
                         original_frames: List[np.ndarray],
                         registered_frames: List[np.ndarray],
                         mask_idx: int) -> Dict:
        """Compute comprehensive quality metrics"""
        
        mask_original = original_frames[mask_idx]
        mask_registered = registered_frames[mask_idx]
        
        per_frame_metrics = []
        
        for i in range(len(registered_frames)):
            if i == mask_idx:
                continue
            
            # Compute metrics
            mi = self.quality_evaluator.compute_mutual_information(
                registered_frames[i],
                mask_registered
            )
            
            ssim_val = self.quality_evaluator.compute_ssim(
                registered_frames[i],
                mask_registered
            )
            
            sharpness = self.quality_evaluator.compute_edge_sharpness(
                registered_frames[i]
            )
            
            per_frame_metrics.append({
                'frame_idx': i,
                'mutual_information': mi,
                'ssim': ssim_val,
                'edge_sharpness': sharpness
            })
        
        # Compute overall statistics
        mi_values = [m['mutual_information'] for m in per_frame_metrics]
        ssim_values = [m['ssim'] for m in per_frame_metrics]
        sharpness_values = [m['edge_sharpness'] for m in per_frame_metrics]
        
        overall_metrics = {
            'mean_mi': np.mean(mi_values),
            'std_mi': np.std(mi_values),
            'mean_ssim': np.mean(ssim_values),
            'std_ssim': np.std(ssim_values),
            'mean_sharpness': np.mean(sharpness_values),
            'std_sharpness': np.std(sharpness_values)
        }
        
        logger.info(f"  Mean MI: {overall_metrics['mean_mi']:.4f} ± {overall_metrics['std_mi']:.4f}")
        logger.info(f"  Mean SSIM: {overall_metrics['mean_ssim']:.4f} ± {overall_metrics['std_ssim']:.4f}")
        logger.info(f"  Mean Sharpness: {overall_metrics['mean_sharpness']:.2f} ± {overall_metrics['std_sharpness']:.2f}")
        
        return {
            'per_frame': per_frame_metrics,
            'overall': overall_metrics
        }


def main_pipeline(input_path: str,
                 output_dir: str,
                 anatomy_type: str = 'peripheral',
                 mask_frame_idx: int = 0,
                 save_dicom: bool = True,
                 create_visualizations: bool = True) -> Dict:
    """
    Complete end-to-end DSA motion correction pipeline
    
    Args:
        input_path: Path to DICOM file or directory
        output_dir: Output directory for results
        anatomy_type: 'neuro', 'peripheral', 'gi', or 'cardiac'
        mask_frame_idx: Index of pre-contrast mask frame
        save_dicom: Save corrected frames as DICOM
        create_visualizations: Generate visualization report
        
    Returns:
        Dictionary with all results
    """
    start_time = datetime.now()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DSA MOTION CORRECTION PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Anatomy: {anatomy_type}")
    logger.info(f"{'='*70}\n")
    
    # Step 1: Load DICOM sequence
    logger.info("Step 1: Loading DICOM sequence...")
    raw_frames, original_datasets = DICOMLoader.load_sequence(Path(input_path))
    logger.info(f"✓ Loaded {len(raw_frames)} frames\n")
    
    # Step 2: Process sequence
    logger.info("Step 2: Processing sequence...")
    processor = DSAProcessor(anatomy_type=anatomy_type)
    results = processor.process_sequence(raw_frames, mask_frame_idx)
    logger.info("✓ Processing complete\n")
    
    # Step 3: Save DICOM outputs
    if save_dicom:
        logger.info("Step 3: Saving DICOM outputs...")
        
        # Save registered frames
        DICOMLoader.save_sequence(
            results['registered_frames'],
            original_datasets,
            output_path / "registered_sequence",
            description=f"DSA Registered - {anatomy_type}"
        )
        
        # Save DSA images
        DICOMLoader.save_sequence(
            results['dsa_images'],
            original_datasets,
            output_path / "dsa_sequence",
            description=f"DSA Subtracted - {anatomy_type}"
        )
        
        logger.info("✓ DICOM sequences saved\n")
    
    # Step 4: Create visualizations
    if create_visualizations:
        logger.info("Step 4: Creating visualizations...")
        try:
            from DAS_motion import visualize_results
            visualize_results(results, output_dir)
            logger.info("✓ Visualizations created\n")
        except ImportError:
            logger.warning("Visualization module not found, skipping...\n")
    
    # Step 5: Save summary report
    logger.info("Step 5: Generating summary report...")
    _save_summary_report(results, output_path / "summary_report.txt")
    logger.info("✓ Summary report saved\n")
    
    # Final summary
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed_time}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frames processed: {len(raw_frames)}")
    logger.info(f"Average SSIM: {results['quality_metrics']['overall']['mean_ssim']:.4f}")
    logger.info(f"Average MI: {results['quality_metrics']['overall']['mean_mi']:.4f}")
    logger.info(f"{'='*70}\n")
    
    return results


def _save_summary_report(results: Dict, output_file: Path):
    """Save text summary report"""
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DSA MOTION CORRECTION - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Anatomy Type: {results['config'].name}\n")
        f.write(f"Total Frames: {len(results['raw_frames'])}\n")
        f.write(f"Mask Frame Index: {results['mask_frame_idx']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFIGURATION PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"ORB Features: {results['config'].orb_features}\n")
        f.write(f"ORB Scale Factor: {results['config'].orb_scale_factor}\n")
        f.write(f"ORB Levels: {results['config'].orb_n_levels}\n")
        f.write(f"RANSAC Threshold: {results['config'].ransac_threshold} pixels\n")
        f.write(f"RANSAC Max Iterations: {results['config'].ransac_max_iter}\n")
        f.write(f"Match Ratio Threshold: {results['config'].match_ratio_threshold}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("REGISTRATION STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Frame':<8} {'Matches':<10} {'Inliers':<10} {'Inlier %':<12}\n")
        f.write("-"*70 + "\n")
        
        for stat in results['registration_stats']:
            if stat['is_mask']:
                f.write(f"{stat['frame_idx']:<8} {'MASK':<10} {'MASK':<10} {'MASK':<12}\n")
            else:
                inlier_pct = (stat['n_inliers'] / stat['n_matches'] * 100) if stat['n_matches'] > 0 else 0
                f.write(f"{stat['frame_idx']:<8} {stat['n_matches']:<10} {stat['n_inliers']:<10} {inlier_pct:<12.1f}\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("QUALITY METRICS\n")
        f.write("-"*70 + "\n")
        
        overall = results['quality_metrics']['overall']
        f.write(f"\nMutual Information:\n")
        f.write(f"  Mean: {overall['mean_mi']:.4f}\n")
        f.write(f"  Std:  {overall['std_mi']:.4f}\n")
        
        f.write(f"\nStructural Similarity (SSIM):\n")
        f.write(f"  Mean: {overall['mean_ssim']:.4f}\n")
        f.write(f"  Std:  {overall['std_ssim']:.4f}\n")
        
        f.write(f"\nEdge Sharpness:\n")
        f.write(f"  Mean: {overall['mean_sharpness']:.2f}\n")
        f.write(f"  Std:  {overall['std_sharpness']:.2f}\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("PER-FRAME QUALITY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Frame':<8} {'MI':<12} {'SSIM':<12} {'Sharpness':<12}\n")
        f.write("-"*70 + "\n")
        
        for metric in results['quality_metrics']['per_frame']:
            f.write(f"{metric['frame_idx']:<8} "
                   f"{metric['mutual_information']:<12.4f} "
                   f"{metric['ssim']:<12.4f} "
                   f"{metric['edge_sharpness']:<12.2f}\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("QUALITY ASSESSMENT\n")
        f.write("-"*70 + "\n")
        
        avg_ssim = overall['mean_ssim']
        if avg_ssim > 0.95:
            quality = "EXCELLENT - Registration quality is outstanding"
        elif avg_ssim > 0.90:
            quality = "GOOD - Registration quality is satisfactory"
        elif avg_ssim > 0.85:
            quality = "ACCEPTABLE - Registration quality is adequate"
        else:
            quality = "POOR - Manual review recommended"
        
        f.write(f"\nOverall Quality: {quality}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("TRANSFORMATION ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        # Analyze transformations
        translations_x = []
        translations_y = []
        rotations = []
        
        for i, matrix in enumerate(results['transformation_matrices']):
            if i == results['mask_frame_idx']:
                continue
            
            tx = matrix[0, 2]
            ty = matrix[1, 2]
            translations_x.append(tx)
            translations_y.append(ty)
            
            rotation = np.arctan2(matrix[1, 0], matrix[0, 0]) * 180 / np.pi
            rotations.append(rotation)
        
        if translations_x:
            f.write(f"\nTranslation X (pixels):\n")
            f.write(f"  Mean: {np.mean(translations_x):.2f}\n")
            f.write(f"  Std:  {np.std(translations_x):.2f}\n")
            f.write(f"  Range: [{np.min(translations_x):.2f}, {np.max(translations_x):.2f}]\n")
            
            f.write(f"\nTranslation Y (pixels):\n")
            f.write(f"  Mean: {np.mean(translations_y):.2f}\n")
            f.write(f"  Std:  {np.std(translations_y):.2f}\n")
            f.write(f"  Range: [{np.min(translations_y):.2f}, {np.max(translations_y):.2f}]\n")
            
            f.write(f"\nRotation (degrees):\n")
            f.write(f"  Mean: {np.mean(rotations):.2f}\n")
            f.write(f"  Std:  {np.std(rotations):.2f}\n")
            f.write(f"  Range: [{np.min(rotations):.2f}, {np.max(rotations):.2f}]\n")
            
            motion_magnitude = np.sqrt(np.array(translations_x)**2 + np.array(translations_y)**2)
            f.write(f"\nMotion Magnitude (pixels):\n")
            f.write(f"  Mean: {np.mean(motion_magnitude):.2f}\n")
            f.write(f"  Max:  {np.max(motion_magnitude):.2f}\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DSA Motion Correction using ORB+RANSAC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process peripheral angiography
  python dsa_motion_correction.py -i /path/to/dicom -o /output -a peripheral
  
  # Process neuro angiography with mask at frame 5
  python dsa_motion_correction.py -i /path/to/dicom -o /output -a neuro -m 5
  
  # Process without visualizations
  python dsa_motion_correction.py -i /path/to/dicom -o /output --no-viz
  
Anatomy types: neuro, peripheral, gi, cardiac
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input DICOM file or directory'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory'
    )
    
    parser.add_argument(
        '-a', '--anatomy',
        default='peripheral',
        choices=['neuro', 'peripheral', 'gi', 'cardiac'],
        help='Anatomy type (default: peripheral)'
    )
    
    parser.add_argument(
        '-m', '--mask-frame',
        type=int,
        default=0,
        help='Index of mask frame (default: 0)'
    )
    
    parser.add_argument(
        '--no-dicom',
        action='store_true',
        help='Skip DICOM output generation'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run pipeline
    try:
        results = main_pipeline(
            input_path=args.input,
            output_dir=args.output,
            anatomy_type=args.anatomy,
            mask_frame_idx=args.mask_frame,
            save_dicom=not args.no_dicom,
            create_visualizations=not args.no_viz
        )
        
        print("\n✓ Processing completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"\n✗ Error during processing: {e}", exc_info=True)
        exit(1)