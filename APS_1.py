import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.optimize as opt
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import time
import warnings
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
except Exception as e:
    raise ImportError("OpenCV is required. Install with `pip install opencv-python`") from e

_has_ximgproc = hasattr(cv2, "ximgproc")

try:
    if hasattr(cv2, "SIFT_create"):
        _SIFT_FACTORY = cv2.SIFT_create
    elif hasattr(cv2, "xfeatures2d"):
        _SIFT_FACTORY = cv2.xfeatures2d.SIFT_create
    else:
        _SIFT_FACTORY = None
except Exception:
    _SIFT_FACTORY = None

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

class MotionModel(Enum):
    """Enumeration of supported motion models"""
    TRANSLATION = "translation"
    RIGID = "rigid"
    SIMILARITY = "similarity"
    AFFINE = "affine"
    PROJECTIVE = "projective"

class RegistrationMethod(Enum):
    """Registration algorithm enumeration"""
    ECC = "ecc"
    ORB_RANSAC = "orb_ransac"
    SIFT_RANSAC = "sift_ransac"
    PHASE_CORRELATION = "phase_corr"
    OPTICAL_FLOW = "optical_flow"
    MI = "mutual_information"
    HYBRID = "hybrid"

@dataclass
class MotionParameters:
    """Data class to store motion parameters and statistics"""
    transform_matrix: np.ndarray
    parameters: np.ndarray
    confidence: float
    convergence_iterations: int
    correlation_coefficient: float
    inlier_ratio: float
    computation_time: float
    method_used: str
    motion_magnitude: float
    angular_displacement: float

@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics"""
    mse: float
    psnr: float
    ssim: float
    ncc: float
    mi: float
    gradient_correlation: float
    vessel_preservation_score: float

class AdvancedDSAMotionCorrection:
    """Advanced Digital Subtraction Angiography Motion Correction System"""
    
    def __init__(self, 
                 motion_model: MotionModel = MotionModel.RIGID,
                 registration_method: RegistrationMethod = RegistrationMethod.ECC,
                 max_iterations: int = 5000,
                 convergence_threshold: float = 1e-10,
                 outlier_threshold: float = 2.0,
                 vessel_detection_sensitivity: float = 2.0,
                 multi_scale_levels: int = 3,
                 enable_gpu_acceleration: bool = False,
                 quality_assessment: bool = True):
        
        self.motion_model = motion_model
        self.registration_method = registration_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.outlier_threshold = outlier_threshold
        self.vessel_detection_sensitivity = vessel_detection_sensitivity
        self.multi_scale_levels = multi_scale_levels
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.quality_assessment = quality_assessment
        
        self._initialize_transform_matrices()
        self._initialize_feature_detectors()
        
        self.performance_stats = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'average_computation_time': 0.0,
            'method_success_rates': {}
        }
        
        logger.info(f"Initialized DSA Motion Correction: {motion_model.value} model")
    
    def _initialize_transform_matrices(self) -> None:
        """Initialize transformation matrices based on motion model"""
        if self.motion_model == MotionModel.TRANSLATION:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 2
        elif self.motion_model == MotionModel.RIGID:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 3
        elif self.motion_model == MotionModel.SIMILARITY:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 4
        elif self.motion_model == MotionModel.AFFINE:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 6
        else:  # PROJECTIVE
            self.transform_matrix = np.eye(3, dtype=np.float32)
            self.dof = 8
    
    def _initialize_feature_detectors(self) -> None:
        """Initialize feature detection algorithms"""
        self.orb_detector = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.15,
            nlevels=12,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=25,
            fastThreshold=15
        )
        
        # Initialize SIFT if available
        if _SIFT_FACTORY is not None:
            try:
                self.sift_detector = _SIFT_FACTORY(
                    nfeatures=1500,
                    nOctaveLayers=4,
                    contrastThreshold=0.03,
                    edgeThreshold=8,
                    sigma=1.4
                )
            except Exception as e:
                logger.warning(f"SIFT initialization failed: {e}")
                self.sift_detector = None
        else:
            self.sift_detector = None
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for SIFT
        if self.sift_detector is not None:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.flann_matcher = None
    
    @staticmethod
    def compute_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
        """
        Compute binary skeleton from vessel mask with multiple fallback methods
        """
        if vessel_mask.dtype != np.uint8:
            vessel_mask = (vessel_mask > 0).astype(np.uint8) * 255

        # Try OpenCV ximgproc thinning
        if _has_ximgproc:
            try:
                sk = cv2.ximgproc.thinning(vessel_mask)
                return (sk > 0).astype(np.uint8) * 255
            except Exception as e:
                logger.debug(f"cv2.ximgproc.thinning failed: {e}")

        # Fall back to skimage
        if skeletonize is not None:
            try:
                sk_bool = skeletonize((vessel_mask > 0))
                return (sk_bool.astype(np.uint8) * 255)
            except Exception as e:
                logger.debug(f"skimage.skeletonize failed: {e}")

        # Last resort: morphological thinning approximation
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
            logger.warning(f"All skeletonization methods failed: {e}")
            return (vessel_mask > 0).astype(np.uint8) * 255
    
    def detect_vessels_advanced(self, 
                               mask_frame: np.ndarray, 
                               live_frame: np.ndarray,
                               method: str = 'multi_criteria') -> Tuple[np.ndarray, Dict]:
        """Advanced vessel detection using multiple criteria"""
        start_time = time.time()
        logger.info(f"Advanced vessel detection using {method} method")
        
        mask_frame = mask_frame.astype(np.float32)
        live_frame = live_frame.astype(np.float32)
        
        vessel_info = {}
        
        if method == 'multi_criteria':
            vessel_mask = self._multi_criteria_vessel_detection(mask_frame, live_frame, vessel_info)
        elif method == 'statistical':
            vessel_mask = self._statistical_vessel_detection(mask_frame, live_frame, vessel_info)
        elif method == 'morphological':
            vessel_mask = self._morphological_vessel_detection(mask_frame, live_frame, vessel_info)
        else:
            raise ValueError(f"Unknown vessel detection method: {method}")
        
        vessel_mask = self._refine_vessel_mask(vessel_mask, vessel_info)
        self._analyze_vessel_characteristics(vessel_mask, live_frame, vessel_info)
        
        vessel_info['computation_time'] = time.time() - start_time
        vessel_info['vessel_pixel_ratio'] = np.sum(vessel_mask > 0) / vessel_mask.size
        
        logger.info(f"Vessel detection completed: {vessel_info['vessel_pixel_ratio']:.3f} ratio")
        
        return vessel_mask, vessel_info
    
    def _multi_criteria_vessel_detection(self, 
                                       mask_frame: np.ndarray, 
                                       live_frame: np.ndarray,
                                       vessel_info: Dict) -> np.ndarray:
        """Multi-criteria vessel detection"""
        
        # Criterion 1: Intensity-based
        mean_live = np.mean(live_frame)
        std_live = np.std(live_frame)
        intensity_threshold = mean_live + self.vessel_detection_sensitivity * std_live
        vessel_mask_intensity = (live_frame > intensity_threshold).astype(np.uint8) * 255
        
        # Criterion 2: Difference-based
        difference = np.abs(live_frame - mask_frame)
        mean_diff = np.mean(difference)
        std_diff = np.std(difference)
        diff_threshold = mean_diff + self.vessel_detection_sensitivity * std_diff
        vessel_mask_diff = (difference > diff_threshold).astype(np.uint8) * 255
        
        # Criterion 3: Gradient-based
        grad_x = cv2.Sobel(live_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(live_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_threshold = np.percentile(gradient_magnitude, 90)
        vessel_mask_grad = (gradient_magnitude > grad_threshold).astype(np.uint8) * 255
        
        # Criterion 4: Local contrast
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(live_frame, -1, kernel)
        local_contrast = live_frame - local_mean
        contrast_threshold = np.std(local_contrast) * 1.5
        vessel_mask_contrast = (local_contrast > contrast_threshold).astype(np.uint8) * 255
        
        # Combine criteria
        weights = [0.3, 0.3, 0.2, 0.2]
        combined_score = (weights[0] * vessel_mask_intensity.astype(np.float32) + 
                         weights[1] * vessel_mask_diff.astype(np.float32) + 
                         weights[2] * vessel_mask_grad.astype(np.float32) + 
                         weights[3] * vessel_mask_contrast.astype(np.float32)) / 255.0
        
        final_threshold = 0.4
        vessel_mask = (combined_score > final_threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'intensity_threshold': float(intensity_threshold),
            'diff_threshold': float(diff_threshold),
            'grad_threshold': float(grad_threshold),
            'contrast_threshold': float(contrast_threshold),
            'final_threshold': final_threshold
        })
        
        return vessel_mask
    
    def _statistical_vessel_detection(self, 
                                    mask_frame: np.ndarray, 
                                    live_frame: np.ndarray,
                                    vessel_info: Dict) -> np.ndarray:
        """Statistical hypothesis testing for vessel detection - optimized version"""
        
        # Use smaller window and stride for efficiency
        window_size = 11  # Reduced from 21
        stride = 5  # Process every 5th pixel for speed
        half_window = window_size // 2
        
        mask_padded = np.pad(mask_frame, half_window, mode='reflect')
        live_padded = np.pad(live_frame, half_window, mode='reflect')
        
        vessel_probability = np.zeros_like(mask_frame)
        
        # Use strided sampling for efficiency
        y_indices = range(half_window, mask_padded.shape[0] - half_window, stride)
        x_indices = range(half_window, mask_padded.shape[1] - half_window, stride)
        
        for i in y_indices:
            for j in x_indices:
                mask_window = mask_padded[i-half_window:i+half_window+1, 
                                        j-half_window:j+half_window+1]
                live_window = live_padded[i-half_window:i+half_window+1, 
                                        j-half_window:j+half_window+1]
                
                try:
                    diff = live_window.flatten() - mask_window.flatten()
                    if np.std(diff) > 1e-6:  # Only test if there's variation
                        statistic, p_value = stats.wilcoxon(diff, alternative='greater')
                        confidence = 1 - p_value if p_value < 0.05 else 0
                    else:
                        confidence = 0
                    vessel_probability[i-half_window, j-half_window] = confidence
                except (ValueError, Warning) as e:
                    logger.debug(f"Statistical test failed at ({i},{j}): {e}")
                    vessel_probability[i-half_window, j-half_window] = \
                        max(0, (np.mean(live_window) - np.mean(mask_window)) / (np.std(mask_window) + 1e-6))
        
        # Interpolate sparse results
        if stride > 1:
            from scipy.interpolate import griddata
            y_sparse, x_sparse = np.meshgrid(
                range(0, mask_frame.shape[0], stride),
                range(0, mask_frame.shape[1], stride),
                indexing='ij'
            )
            y_dense, x_dense = np.meshgrid(
                range(mask_frame.shape[0]),
                range(mask_frame.shape[1]),
                indexing='ij'
            )
            points = np.column_stack([y_sparse.ravel(), x_sparse.ravel()])
            values = vessel_probability[::stride, ::stride].ravel()
            vessel_probability = griddata(points, values, (y_dense, x_dense), method='linear', fill_value=0)
        
        probability_threshold = 0.7
        vessel_mask = (vessel_probability > probability_threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'statistical_method': 'wilcoxon_signed_rank',
            'probability_threshold': probability_threshold,
            'mean_probability': float(np.mean(vessel_probability)),
            'max_probability': float(np.max(vessel_probability))
        })
        
        return vessel_mask
    
    def _morphological_vessel_detection(self, 
                                      mask_frame: np.ndarray, 
                                      live_frame: np.ndarray,
                                      vessel_info: Dict) -> np.ndarray:
        """Morphology-based vessel detection"""
        
        vessel_responses = []
        scales = [3, 5, 7, 9, 11]
        
        for scale in scales:
            kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, max(1, scale//3)))
            kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
            
            tophat_line = cv2.morphologyEx(live_frame, cv2.MORPH_TOPHAT, kernel_line)
            tophat_circle = cv2.morphologyEx(live_frame, cv2.MORPH_TOPHAT, kernel_circle)
            
            vessel_response = np.maximum(tophat_line, tophat_circle)
            vessel_responses.append(vessel_response)
        
        combined_response = np.maximum.reduce(vessel_responses)
        
        threshold = np.mean(combined_response) + 2 * np.std(combined_response)
        vessel_mask = (combined_response > threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'morphological_scales': scales,
            'threshold': float(threshold),
            'max_response': float(np.max(combined_response))
        })
        
        return vessel_mask
    
    def _refine_vessel_mask(self, vessel_mask: np.ndarray, vessel_info: Dict) -> np.ndarray:
        """Refine vessel mask using morphological operations"""
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_open)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vessel_mask)
        
        min_area = 50
        max_aspect_ratio = 10
        
        refined_mask = np.zeros_like(vessel_mask)
        valid_components = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area >= min_area:
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if aspect_ratio <= max_aspect_ratio:
                    refined_mask[labels == i] = 255
                    valid_components += 1
        
        vessel_info.update({
            'total_components': num_labels - 1,
            'valid_components': valid_components
        })
        
        return refined_mask
    
    def _analyze_vessel_characteristics(self, 
                                      vessel_mask: np.ndarray, 
                                      live_frame: np.ndarray,
                                      vessel_info: Dict) -> None:
        """Analyze vessel characteristics"""
        
        if np.sum(vessel_mask) == 0:
            vessel_info['vessel_characteristics'] = {}
            return
        
        vessel_pixels = live_frame[vessel_mask > 0]
        
        characteristics = {
            'mean_intensity': float(np.mean(vessel_pixels)),
            'std_intensity': float(np.std(vessel_pixels)),
            'min_intensity': float(np.min(vessel_pixels)),
            'max_intensity': float(np.max(vessel_pixels)),
            'total_area': int(np.sum(vessel_mask > 0)),
            'intensity_range': float(np.max(vessel_pixels) - np.min(vessel_pixels))
        }
        
        # Compute skeleton safely
        try:
            skeleton = self.compute_skeleton(vessel_mask)
            characteristics['skeleton_length'] = int(np.sum(skeleton > 0))
            characteristics['tortuosity'] = float(characteristics['skeleton_length'] / (characteristics['total_area'] + 1e-6))
        except Exception as e:
            logger.warning(f"Skeleton computation failed: {e}")
            characteristics['skeleton_length'] = 0
            characteristics['tortuosity'] = 0.0
        
        vessel_info['vessel_characteristics'] = characteristics
    
    def create_registration_mask_advanced(self, 
                                        vessel_mask: np.ndarray,
                                        mask_frame: np.ndarray,
                                        method: str = 'adaptive') -> Tuple[np.ndarray, Dict]:
        """Advanced registration mask creation"""
        logger.info(f"Creating registration mask using {method} method")
        start_time = time.time()
        
        mask_info = {}
        
        if method == 'adaptive':
            registration_mask = self._adaptive_registration_mask(vessel_mask, mask_frame, mask_info)
        elif method == 'gradient_weighted':
            registration_mask = self._gradient_weighted_mask(vessel_mask, mask_frame, mask_info)
        else:
            registration_mask = cv2.bitwise_not(vessel_mask)
            
        mask_info['computation_time'] = time.time() - start_time
        mask_info['registration_pixel_ratio'] = float(np.sum(registration_mask > 0) / registration_mask.size)
        
        return registration_mask, mask_info
    
    def _adaptive_registration_mask(self, 
                                  vessel_mask: np.ndarray, 
                                  mask_frame: np.ndarray,
                                  mask_info: Dict) -> np.ndarray:
        """Create adaptive registration mask"""
        
        vessel_distance = cv2.distanceTransform(cv2.bitwise_not(vessel_mask), cv2.DIST_L2, 5)
        
        grad_x = cv2.Sobel(mask_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_reliability = gradient_magnitude / (np.max(gradient_magnitude) + 1e-6)
        
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(mask_frame, -1, kernel)
        local_variance = cv2.filter2D(mask_frame**2, -1, kernel) - local_mean**2
        variance_reliability = local_variance / (np.max(local_variance) + 1e-6)
        
        h, w = mask_frame.shape
        y, x = np.ogrid[:h, :w]
        border_distance = np.minimum(np.minimum(x, w-x), np.minimum(y, h-y))
        border_reliability = np.minimum(border_distance / 50.0, 1.0)
        
        combined_reliability = (0.4 * gradient_reliability + 
                              0.4 * variance_reliability + 
                              0.2 * border_reliability)
        
        spatial_weight = np.exp(-vessel_distance / 20.0)
        registration_weight = combined_reliability * spatial_weight
        
        weight_threshold = np.percentile(registration_weight, 30)
        registration_mask = (registration_weight > weight_threshold).astype(np.uint8) * 255
        registration_mask = cv2.bitwise_and(registration_mask, cv2.bitwise_not(vessel_mask))
        
        mask_info.update({
            'weight_threshold': float(weight_threshold),
            'mean_weight': float(np.mean(registration_weight))
        })
        
        return registration_mask
    
    def _gradient_weighted_mask(self, 
                               vessel_mask: np.ndarray, 
                               mask_frame: np.ndarray,
                               mask_info: Dict) -> np.ndarray:
        """Create gradient-weighted registration mask"""
        
        grad_x = cv2.Sobel(mask_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_threshold = np.percentile(gradient_magnitude, 50)
        registration_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255
        registration_mask = cv2.bitwise_and(registration_mask, cv2.bitwise_not(vessel_mask))
        
        mask_info['gradient_threshold'] = float(gradient_threshold)
        
        return registration_mask
    
    def estimate_motion_ecc_advanced(self, 
                                   mask_frame: np.ndarray, 
                                   live_frame: np.ndarray,
                                   registration_mask: Optional[np.ndarray] = None,
                                   initial_transform: Optional[np.ndarray] = None) -> MotionParameters:
        """Advanced ECC motion estimation"""
        logger.info("Advanced ECC motion estimation")
        start_time = time.time()
        
        if initial_transform is not None:
            transform_matrix = initial_transform.copy()
        else:
            transform_matrix = self.transform_matrix.copy()
        
        cv_motion_types = {
            MotionModel.TRANSLATION: cv2.MOTION_TRANSLATION,
            MotionModel.RIGID: cv2.MOTION_EUCLIDEAN,
            MotionModel.AFFINE: cv2.MOTION_AFFINE
        }
        
        if self.motion_model not in cv_motion_types:
            logger.warning(f"Motion model {self.motion_model} not supported by ECC, using AFFINE")
            motion_type = cv2.MOTION_AFFINE
        else:
            motion_type = cv_motion_types[self.motion_model]
        
        correlation_coeffs = []
        best_cc = -1
        best_transform = transform_matrix.copy()
        
        for level in range(self.multi_scale_levels, 0, -1):
            scale_factor = 2**(level-1)
            if scale_factor > 1:
                scaled_mask = cv2.resize(mask_frame, None, fx=1/scale_factor, fy=1/scale_factor)
                scaled_live = cv2.resize(live_frame, None, fx=1/scale_factor, fy=1/scale_factor)
                scaled_reg_mask = cv2.resize(registration_mask, None, 
                                           fx=1/scale_factor, fy=1/scale_factor) if registration_mask is not None else None
                scaled_transform = transform_matrix.copy()
                scaled_transform[:, 2] /= scale_factor
            else:
                scaled_mask = mask_frame
                scaled_live = live_frame
                scaled_reg_mask = registration_mask
                scaled_transform = transform_matrix
            
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                       self.max_iterations // self.multi_scale_levels, 
                       self.convergence_threshold)
            
            try:
                if scaled_reg_mask is not None:
                    cc, level_transform = cv2.findTransformECC(
                        scaled_mask, scaled_live, scaled_transform,
                        motion_type, criteria, scaled_reg_mask
                    )
                else:
                    cc, level_transform = cv2.findTransformECC(
                        scaled_mask, scaled_live, scaled_transform,
                        motion_type, criteria
                    )
                
                correlation_coeffs.append(cc)
                
                if scale_factor > 1:
                    level_transform[:, 2] *= scale_factor
                
                if cc > best_cc:
                    best_cc = cc
                    best_transform = level_transform.copy()
                
                transform_matrix = level_transform.copy()
                logger.info(f"Level {level}: CC = {cc:.6f}")
                
            except cv2.error as e:
                logger.warning(f"ECC failed at level {level}: {e}")
                correlation_coeffs.append(-1)
        
        parameters = self._extract_motion_parameters(best_transform)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        motion_params = MotionParameters(
            transform_matrix=best_transform,
            parameters=parameters,
            confidence=best_cc,
            convergence_iterations=len([cc for cc in correlation_coeffs if cc > 0]),
            correlation_coefficient=best_cc,
            inlier_ratio=1.0,
            computation_time=time.time() - start_time,
            method_used="ECC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"ECC completed: CC={best_cc:.6f}, Motion={motion_magnitude:.2f}px")
        
        return motion_params
    
    def estimate_motion_orb_ransac_advanced(self, 
                                          mask_frame: np.ndarray, 
                                          live_frame: np.ndarray,
                                          registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """Advanced ORB+RANSAC motion estimation"""
        logger.info("Advanced ORB+RANSAC motion estimation")
        start_time = time.time()
        
        kp1, des1 = self.orb_detector.detectAndCompute(mask_frame, registration_mask)
        kp2, des2 = self.orb_detector.detectAndCompute(live_frame, registration_mask)
        
        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            logger.warning("Insufficient features detected")
            return self._create_identity_motion_parameters("ORB_insufficient_features")
        
        matches = self._advanced_feature_matching(des1, des2, kp1, kp2)
        
        if len(matches) < 10:
            logger.warning("Insufficient good matches")
            return self._create_identity_motion_parameters("ORB_insufficient_matches")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        transform_matrix, inliers = self._advanced_ransac_estimation(src_pts, dst_pts)
        
        if transform_matrix is None:
            logger.warning("RANSAC estimation failed")
            return self._create_identity_motion_parameters("ORB_ransac_failed")
        
        validation_score = self._validate_transformation(src_pts, dst_pts, transform_matrix, inliers)
        
        parameters = self._extract_motion_parameters(transform_matrix)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        inlier_ratio = np.sum(inliers) / len(matches)
        
        motion_params = MotionParameters(
            transform_matrix=transform_matrix,
            parameters=parameters,
            confidence=validation_score,
            convergence_iterations=1,
            correlation_coefficient=validation_score,
            inlier_ratio=inlier_ratio,
            computation_time=time.time() - start_time,
            method_used="ORB_RANSAC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"ORB+RANSAC completed: Features={len(kp1)}/{len(kp2)}, Matches={len(matches)}, Inliers={np.sum(inliers)}")
        
        return motion_params
    
    def estimate_motion_sift_ransac_advanced(self,
                                           mask_frame: np.ndarray, 
                                           live_frame: np.ndarray,
                                           registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """SIFT-based motion estimation"""
        if self.sift_detector is None:
            logger.warning("SIFT not available, falling back to ORB")
            return self.estimate_motion_orb_ransac_advanced(mask_frame, live_frame, registration_mask)
        
        logger.info("SIFT+RANSAC motion estimation")
        start_time = time.time()
        
        kp1, des1 = self.sift_detector.detectAndCompute(mask_frame, registration_mask)
        kp2, des2 = self.sift_detector.detectAndCompute(live_frame, registration_mask)
        
        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            logger.warning("Insufficient SIFT features detected")
            return self._create_identity_motion_parameters("SIFT_insufficient_features")
        
        # FLANN matching for SIFT
        matches = self.flann_matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            logger.warning("Insufficient SIFT matches")
            return self._create_identity_motion_parameters("SIFT_insufficient_matches")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        transform_matrix, inliers = self._advanced_ransac_estimation(src_pts, dst_pts)
        
        if transform_matrix is None:
            logger.warning("SIFT RANSAC failed")
            return self._create_identity_motion_parameters("SIFT_ransac_failed")
        
        validation_score = self._validate_transformation(src_pts, dst_pts, transform_matrix, inliers)
        
        parameters = self._extract_motion_parameters(transform_matrix)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        inlier_ratio = np.sum(inliers) / len(good_matches)
        
        motion_params = MotionParameters(
            transform_matrix=transform_matrix,
            parameters=parameters,
            confidence=validation_score,
            convergence_iterations=1,
            correlation_coefficient=validation_score,
            inlier_ratio=inlier_ratio,
            computation_time=time.time() - start_time,
            method_used="SIFT_RANSAC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"SIFT+RANSAC completed: Matches={len(good_matches)}, Inliers={np.sum(inliers)}")
        
        return motion_params
    
    def estimate_motion_phase_correlation(self,
                                        mask_frame: np.ndarray, 
                                        live_frame: np.ndarray) -> MotionParameters:
        """Phase correlation motion estimation for translation"""
        logger.info("Phase correlation motion estimation")
        start_time = time.time()
        
        # Phase correlation only works for translation
        try:
            # Convert to proper format
            mask_32f = mask_frame.astype(np.float32)
            live_32f = live_frame.astype(np.float32)
            
            # Compute phase correlation
            shift, response = cv2.phaseCorrelate(mask_32f, live_32f)
            
            tx, ty = shift
            
            # Create translation matrix
            if self.motion_model == MotionModel.TRANSLATION:
                transform_matrix = np.array([[1.0, 0.0, tx],
                                           [0.0, 1.0, ty]], dtype=np.float32)
            else:
                # Extend to requested model with translation only
                transform_matrix = self.transform_matrix.copy()
                transform_matrix[0, 2] = tx
                transform_matrix[1, 2] = ty
            
            parameters = self._extract_motion_parameters(transform_matrix)
            motion_magnitude = np.linalg.norm([tx, ty])
            
            # Response is the confidence measure
            confidence = float(response)
            
            motion_params = MotionParameters(
                transform_matrix=transform_matrix,
                parameters=parameters,
                confidence=confidence,
                convergence_iterations=1,
                correlation_coefficient=confidence,
                inlier_ratio=1.0,
                computation_time=time.time() - start_time,
                method_used="Phase_Correlation",
                motion_magnitude=motion_magnitude,
                angular_displacement=0.0
            )
            
            logger.info(f"Phase correlation completed: shift=({tx:.2f}, {ty:.2f}), response={confidence:.3f}")
            
            return motion_params
            
        except Exception as e:
            logger.error(f"Phase correlation failed: {e}")
            return self._create_identity_motion_parameters("Phase_Correlation_failed")
    
    def estimate_motion_hybrid(self,
                            mask_frame: np.ndarray, 
                            live_frame: np.ndarray,
                            registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """Hybrid motion estimation combining multiple methods"""
        logger.info("Hybrid motion estimation - combining multiple methods")
        start_time = time.time()
        
        # Try multiple methods
        methods_results = []
        
        # Method 1: ECC
        try:
            ecc_params = self.estimate_motion_ecc_advanced(mask_frame, live_frame, registration_mask)
            methods_results.append(('ECC', ecc_params))
            logger.info(f"ECC confidence: {ecc_params.confidence:.3f}")
        except Exception as e:
            logger.warning(f"ECC failed: {e}")
        
        # Method 2: ORB
        try:
            orb_params = self.estimate_motion_orb_ransac_advanced(mask_frame, live_frame, registration_mask)
            methods_results.append(('ORB', orb_params))
            logger.info(f"ORB confidence: {orb_params.confidence:.3f}")
        except Exception as e:
            logger.warning(f"ORB failed: {e}")
        
        # Method 3: Phase Correlation (if translation model)
        if self.motion_model == MotionModel.TRANSLATION:
            try:
                phase_params = self.estimate_motion_phase_correlation(mask_frame, live_frame)
                methods_results.append(('Phase', phase_params))
                logger.info(f"Phase correlation confidence: {phase_params.confidence:.3f}")
            except Exception as e:
                logger.warning(f"Phase correlation failed: {e}")
        
        if not methods_results:
            logger.error("All hybrid methods failed")
            return self._create_identity_motion_parameters("Hybrid_all_failed")
        
        # Select best result based on confidence
        best_method, best_params = max(methods_results, key=lambda x: x[1].confidence)
        
        # Update method name
        best_params.method_used = f"Hybrid_{best_method}_selected"
        best_params.computation_time = time.time() - start_time
        
        logger.info(f"Hybrid selected: {best_method} with confidence {best_params.confidence:.3f}")
        
        return best_params
    
    def _advanced_feature_matching(self, 
                                 des1: np.ndarray, 
                                 des2: np.ndarray,
                                 kp1: List, 
                                 kp2: List) -> List:
        """Advanced feature matching with validation"""
        
        matches = self.bf_matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
                
            m, n = match_pair
            
            if m.distance < 0.75 * n.distance:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                
                displacement = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if displacement < 200:
                    scale1 = kp1[m.queryIdx].size
                    scale2 = kp2[m.trainIdx].size
                    scale_ratio = max(scale1, scale2) / (min(scale1, scale2) + 1e-6)
                    
                    if scale_ratio < 2.0:
                        good_matches.append(m)
        
        # Spatial clustering
        if len(good_matches) > 20:
            match_coords = []
            for m in good_matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                displacement_vec = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
                match_coords.append(displacement_vec)
            
            match_coords = np.array(match_coords)
            try:
                clustering = DBSCAN(eps=10, min_samples=3).fit(match_coords)
                
                if len(np.unique(clustering.labels_)) > 1:
                    valid_labels = clustering.labels_[clustering.labels_ >= 0]
                    if len(valid_labels) > 0:
                        largest_cluster = np.bincount(valid_labels).argmax()
                        cluster_mask = clustering.labels_ == largest_cluster
                        good_matches = [good_matches[i] for i in range(len(good_matches)) if cluster_mask[i]]
            except Exception as e:
                logger.debug(f"Clustering failed: {e}")
        
        return good_matches
    
    def _advanced_ransac_estimation(self, 
                                  src_pts: np.ndarray, 
                                  dst_pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advanced RANSAC estimation with adaptive parameters"""
        
        num_points = len(src_pts)
        confidence = 0.995
        
        initial_inlier_ratio = min(0.8, max(0.3, num_points / 100))
        
        min_samples = 3 if self.motion_model == MotionModel.RIGID else 4
        max_iterations = min(5000, int(np.log(1 - confidence) / 
                                    np.log(1 - initial_inlier_ratio**min_samples + 1e-10)))
        
        best_transform = None
        best_inliers = np.array([])
        max_inliers = 0
        
        thresholds = [self.outlier_threshold * 2, self.outlier_threshold, self.outlier_threshold * 0.5]
        
        for threshold in thresholds:
            if self.motion_model == MotionModel.TRANSLATION:
                transform, inliers = self._estimate_translation_ransac(
                    src_pts, dst_pts, threshold, max_iterations
                )
            elif self.motion_model == MotionModel.RIGID:
                transform, inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=threshold,
                    maxIters=max_iterations,
                    confidence=confidence
                )
            else:
                transform, inliers = cv2.estimateAffine2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=threshold,
                    maxIters=max_iterations,
                    confidence=confidence
                )
            
            if transform is not None and inliers is not None:
                num_inliers = np.sum(inliers)
                if num_inliers > max_inliers:
                    best_transform = transform
                    best_inliers = inliers.flatten()
                    max_inliers = num_inliers
        
        return best_transform, best_inliers
    
    def _estimate_translation_ransac(self, 
                                   src_pts: np.ndarray, 
                                   dst_pts: np.ndarray,
                                   threshold: float, 
                                   max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Custom RANSAC for translation-only model"""
        
        best_transform = None
        best_inliers = np.array([])
        max_inlier_count = 0
        
        num_pts = len(src_pts)
        if num_pts < 1:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), np.array([])
        
        for _ in range(max_iterations):
            sample_idx = np.random.choice(num_pts, 1, replace=False)
            
            translation = dst_pts[sample_idx] - src_pts[sample_idx]
            tx, ty = translation[0, 0]
            
            transform = np.array([[1.0, 0.0, tx],
                                [0.0, 1.0, ty]], dtype=np.float32)
            
            transformed_pts = src_pts + translation
            distances = np.linalg.norm(dst_pts - transformed_pts, axis=2).flatten()
            inliers = distances < threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_transform = transform
                best_inliers = inliers
        
        return best_transform, best_inliers
    
    def _validate_transformation(self, 
                               src_pts: np.ndarray, 
                               dst_pts: np.ndarray,
                               transform: np.ndarray, 
                               inliers: np.ndarray) -> float:
        """Statistical validation of estimated transformation"""
        
        if len(inliers) == 0 or np.sum(inliers) < 3:
            return 0.0
        
        src_homogeneous = np.hstack([src_pts.reshape(-1, 2), np.ones((len(src_pts), 1))])
        transformed_pts = (transform @ src_homogeneous.T).T
        
        inlier_mask = inliers.astype(bool)
        residuals = np.linalg.norm(dst_pts.reshape(-1, 2)[inlier_mask] - 
                                 transformed_pts[inlier_mask], axis=1)
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        inlier_ratio = np.sum(inliers) / len(inliers)
        residual_score = max(0, 1 - mean_residual / 10.0)
        stability_score = max(0, 1 - std_residual / 5.0)
        
        validation_score = (0.5 * inlier_ratio + 
                          0.3 * residual_score + 
                          0.2 * stability_score)
        
        return validation_score
    
    def _extract_motion_parameters(self, transform_matrix: np.ndarray) -> np.ndarray:
        """Extract motion parameters from transformation matrix"""
        
        if self.motion_model == MotionModel.TRANSLATION:
            return np.array([transform_matrix[0, 2], transform_matrix[1, 2]])
        
        elif self.motion_model == MotionModel.RIGID:
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            return np.array([tx, ty, angle])
        
        elif self.motion_model == MotionModel.SIMILARITY:
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            scale = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            return np.array([tx, ty, angle, scale])
        
        else:  # AFFINE
            return transform_matrix.flatten()[:6]
    
    def _create_identity_motion_parameters(self, method: str) -> MotionParameters:
        """Create identity motion parameters for failed cases"""
        return MotionParameters(
            transform_matrix=self.transform_matrix.copy(),
            parameters=np.zeros(self.dof),
            confidence=0.0,
            convergence_iterations=0,
            correlation_coefficient=0.0,
            inlier_ratio=0.0,
            computation_time=0.0,
            method_used=method,
            motion_magnitude=0.0,
            angular_displacement=0.0
        )
    
    def apply_transformation_advanced(self, 
                                    live_frame: np.ndarray,
                                    motion_params: MotionParameters,
                                    vessel_mask: Optional[np.ndarray] = None,
                                    preservation_method: str = 'selective') -> Tuple[np.ndarray, Dict]:
        """Advanced transformation application with vessel preservation"""
        logger.info(f"Applying transformation using {preservation_method} preservation")
        start_time = time.time()
        
        transform_info = {}
        
        if self.motion_model == MotionModel.PROJECTIVE:
            h, w = live_frame.shape[:2]
            aligned_frame = cv2.warpPerspective(live_frame, motion_params.transform_matrix, 
                                              (w, h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)
        else:
            aligned_frame = cv2.warpAffine(live_frame, motion_params.transform_matrix, 
                                         (live_frame.shape[1], live_frame.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
        
        if vessel_mask is not None and preservation_method != 'none':
            if preservation_method == 'selective':
                aligned_frame = self._selective_vessel_preservation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
            elif preservation_method == 'blended':
                aligned_frame = self._blended_vessel_preservation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
            elif preservation_method == 'intensity_preserved':
                aligned_frame = self._intensity_preserved_transformation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
        
        transform_info['computation_time'] = time.time() - start_time
        transform_info['preservation_method'] = preservation_method
        
        return aligned_frame, transform_info
    
    def _selective_vessel_preservation(self, 
                                     original: np.ndarray,
                                     warped: np.ndarray, 
                                     vessel_mask: np.ndarray,
                                     transform_info: Dict) -> np.ndarray:
        """Selectively preserve vessel pixels"""
        
        result = warped.copy()
        vessel_pixels = vessel_mask > 0
        
        result[vessel_pixels] = original[vessel_pixels]
        
        transform_info['preserved_pixels'] = int(np.sum(vessel_pixels))
        transform_info['preservation_ratio'] = float(np.sum(vessel_pixels) / vessel_mask.size)
        
        return result
    
    def _blended_vessel_preservation(self, 
                                   original: np.ndarray,
                                   warped: np.ndarray, 
                                   vessel_mask: np.ndarray,
                                   transform_info: Dict) -> np.ndarray:
        """Smooth blending at vessel boundaries - FIXED"""
        
        vessel_distance = cv2.distanceTransform(cv2.bitwise_not(vessel_mask), cv2.DIST_L2, 5)
        
        blend_distance = 5.0
        weights = 1.0 / (1.0 + np.exp(-(vessel_distance - blend_distance) / 2.0))
        
        # Fixed dimension handling
        if len(original.shape) == 2:
            result = weights * warped + (1 - weights) * original
        else:
            result = weights[..., np.newaxis] * warped + (1 - weights[..., np.newaxis]) * original
        
        transform_info['blend_distance'] = blend_distance
        transform_info['blended_pixels'] = int(np.sum(weights < 0.99))
        
        return result.astype(original.dtype)
    
    def _intensity_preserved_transformation(self, 
                                          original: np.ndarray,
                                          warped: np.ndarray, 
                                          vessel_mask: np.ndarray,
                                          transform_info: Dict) -> np.ndarray:
        """Preserve vessel intensity relationships"""
        
        vessel_pixels = vessel_mask > 0
        
        if np.sum(vessel_pixels) == 0:
            return warped
        
        original_vessel_mean = np.mean(original[vessel_pixels])
        warped_vessel_mean = np.mean(warped[vessel_pixels])
        
        if warped_vessel_mean > 0:
            intensity_scale = original_vessel_mean / warped_vessel_mean
        else:
            intensity_scale = 1.0
        
        result = warped.copy()
        result[vessel_pixels] = np.clip(warped[vessel_pixels] * intensity_scale, 
                                       0, 255).astype(original.dtype)
        
        transform_info['intensity_scale_factor'] = float(intensity_scale)
        transform_info['corrected_pixels'] = int(np.sum(vessel_pixels))
        
        return result
    
    def compute_quality_metrics(self, 
                              mask_frame: np.ndarray,
                              original_live: np.ndarray, 
                              aligned_live: np.ndarray,
                              vessel_mask: Optional[np.ndarray] = None) -> ImageQualityMetrics:
        """Comprehensive image quality assessment"""
        logger.info("Computing quality metrics")
        
        mask_frame = mask_frame.astype(np.float32)
        original_live = original_live.astype(np.float32)
        aligned_live = aligned_live.astype(np.float32)
        
        # MSE
        mse = float(np.mean((mask_frame - aligned_live)**2))
        
        # PSNR
        max_intensity = 255.0
        psnr = float(20 * np.log10(max_intensity / (np.sqrt(mse) + 1e-8)))
        
        # SSIM
        ssim = self._compute_ssim(mask_frame, aligned_live)
        
        # NCC
        ncc = self._compute_ncc(mask_frame, aligned_live)
        
        # Mutual Information
        mi = self._compute_mutual_information(mask_frame, aligned_live)
        
        # Gradient Correlation
        grad_corr = self._compute_gradient_correlation(mask_frame, aligned_live)
        
        # Vessel preservation
        if vessel_mask is not None:
            vessel_preservation = self._compute_vessel_preservation_score(
                original_live, aligned_live, vessel_mask
            )
        else:
            vessel_preservation = 1.0
        
        return ImageQualityMetrics(
            mse=mse,
            psnr=psnr,
            ssim=ssim,
            ncc=ncc,
            mi=mi,
            gradient_correlation=grad_corr,
            vessel_preservation_score=vessel_preservation
        )
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM"""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.T)
        
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    def _compute_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute NCC"""
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        
        numerator = np.sum((img1 - mean1) * (img2 - mean2))
        denominator = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
        
        return float(numerator / (denominator + 1e-8))
    
    def _compute_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute MI"""
        img1_discrete = np.round(img1).astype(int)
        img2_discrete = np.round(img2).astype(int)
        
        joint_hist, _, _ = np.histogram2d(img1_discrete.flatten(), img2_discrete.flatten(), 
                                        bins=256, range=[[0, 255], [0, 255]])
        joint_hist = joint_hist + 1e-10
        
        joint_prob = joint_hist / np.sum(joint_hist)
        
        marginal1 = np.sum(joint_prob, axis=1)
        marginal2 = np.sum(joint_prob, axis=0)
        
        mi = 0.0
        for i in range(256):
            for j in range(256):
                if joint_prob[i, j] > 1e-10:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / 
                                                   (marginal1[i] * marginal2[j] + 1e-10))
        
        return float(mi)
    
    def _compute_gradient_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute gradient correlation"""
        grad1_x = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        return self._compute_ncc(grad1_mag, grad2_mag)
    
    def _compute_vessel_preservation_score(self, 
                                         original: np.ndarray,
                                         aligned: np.ndarray, 
                                         vessel_mask: np.ndarray) -> float:
        """Compute vessel preservation score"""
        vessel_pixels = vessel_mask > 0
        
        if np.sum(vessel_pixels) == 0:
            return 1.0
        
        original_vessels = original[vessel_pixels]
        aligned_vessels = aligned[vessel_pixels]
        
        vessel_correlation = self._compute_ncc(original_vessels, aligned_vessels)
        
        intensity_ratio = np.mean(aligned_vessels) / (np.mean(original_vessels) + 1e-8)
        intensity_preservation = 1.0 - abs(1.0 - intensity_ratio)
        
        return float(0.7 * vessel_correlation + 0.3 * intensity_preservation)
    
    def correct_motion_comprehensive(self,import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.optimize as opt
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import time
import warnings
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
except Exception as e:
    raise ImportError("OpenCV is required. Install with `pip install opencv-python`") from e

_has_ximgproc = hasattr(cv2, "ximgproc")

try:
    if hasattr(cv2, "SIFT_create"):
        _SIFT_FACTORY = cv2.SIFT_create
    elif hasattr(cv2, "xfeatures2d"):
        _SIFT_FACTORY = cv2.xfeatures2d.SIFT_create
    else:
        _SIFT_FACTORY = None
except Exception:
    _SIFT_FACTORY = None

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

class MotionModel(Enum):
    """Enumeration of supported motion models"""
    TRANSLATION = "translation"
    RIGID = "rigid"
    SIMILARITY = "similarity"
    AFFINE = "affine"
    PROJECTIVE = "projective"

class RegistrationMethod(Enum):
    """Registration algorithm enumeration"""
    ECC = "ecc"
    ORB_RANSAC = "orb_ransac"
    SIFT_RANSAC = "sift_ransac"
    PHASE_CORRELATION = "phase_corr"
    OPTICAL_FLOW = "optical_flow"
    MI = "mutual_information"
    HYBRID = "hybrid"

@dataclass
class MotionParameters:
    """Data class to store motion parameters and statistics"""
    transform_matrix: np.ndarray
    parameters: np.ndarray
    confidence: float
    convergence_iterations: int
    correlation_coefficient: float
    inlier_ratio: float
    computation_time: float
    method_used: str
    motion_magnitude: float
    angular_displacement: float

@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics"""
    mse: float
    psnr: float
    ssim: float
    ncc: float
    mi: float
    gradient_correlation: float
    vessel_preservation_score: float

class AdvancedDSAMotionCorrection:
    """Advanced Digital Subtraction Angiography Motion Correction System"""
    
    def __init__(self, 
                 motion_model: MotionModel = MotionModel.RIGID,
                 registration_method: RegistrationMethod = RegistrationMethod.ECC,
                 max_iterations: int = 5000,
                 convergence_threshold: float = 1e-10,
                 outlier_threshold: float = 2.0,
                 vessel_detection_sensitivity: float = 2.0,
                 multi_scale_levels: int = 3,
                 enable_gpu_acceleration: bool = False,
                 quality_assessment: bool = True):
        
        self.motion_model = motion_model
        self.registration_method = registration_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.outlier_threshold = outlier_threshold
        self.vessel_detection_sensitivity = vessel_detection_sensitivity
        self.multi_scale_levels = multi_scale_levels
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.quality_assessment = quality_assessment
        
        self._initialize_transform_matrices()
        self._initialize_feature_detectors()
        
        self.performance_stats = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'average_computation_time': 0.0,
            'method_success_rates': {}
        }
        
        logger.info(f"Initialized DSA Motion Correction: {motion_model.value} model")
    
    def _initialize_transform_matrices(self) -> None:
        """Initialize transformation matrices based on motion model"""
        if self.motion_model == MotionModel.TRANSLATION:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 2
        elif self.motion_model == MotionModel.RIGID:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 3
        elif self.motion_model == MotionModel.SIMILARITY:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 4
        elif self.motion_model == MotionModel.AFFINE:
            self.transform_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32)
            self.dof = 6
        else:  # PROJECTIVE
            self.transform_matrix = np.eye(3, dtype=np.float32)
            self.dof = 8
    
    def _initialize_feature_detectors(self) -> None:
        """Initialize feature detection algorithms"""
        self.orb_detector = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.15,
            nlevels=12,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=25,
            fastThreshold=15
        )
        
        # Initialize SIFT if available
        if _SIFT_FACTORY is not None:
            try:
                self.sift_detector = _SIFT_FACTORY(
                    nfeatures=1500,
                    nOctaveLayers=4,
                    contrastThreshold=0.03,
                    edgeThreshold=8,
                    sigma=1.4
                )
            except Exception as e:
                logger.warning(f"SIFT initialization failed: {e}")
                self.sift_detector = None
        else:
            self.sift_detector = None
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for SIFT
        if self.sift_detector is not None:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.flann_matcher = None
    
    @staticmethod
    def compute_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
        """
        Compute binary skeleton from vessel mask with multiple fallback methods
        """
        if vessel_mask.dtype != np.uint8:
            vessel_mask = (vessel_mask > 0).astype(np.uint8) * 255

        # Try OpenCV ximgproc thinning
        if _has_ximgproc:
            try:
                sk = cv2.ximgproc.thinning(vessel_mask)
                return (sk > 0).astype(np.uint8) * 255
            except Exception as e:
                logger.debug(f"cv2.ximgproc.thinning failed: {e}")

        # Fall back to skimage
        if skeletonize is not None:
            try:
                sk_bool = skeletonize((vessel_mask > 0))
                return (sk_bool.astype(np.uint8) * 255)
            except Exception as e:
                logger.debug(f"skimage.skeletonize failed: {e}")

        # Last resort: morphological thinning approximation
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
            logger.warning(f"All skeletonization methods failed: {e}")
            return (vessel_mask > 0).astype(np.uint8) * 255
    
    def detect_vessels_advanced(self, 
                               mask_frame: np.ndarray, 
                               live_frame: np.ndarray,
                               method: str = 'multi_criteria') -> Tuple[np.ndarray, Dict]:
        """Advanced vessel detection using multiple criteria"""
        start_time = time.time()
        logger.info(f"Advanced vessel detection using {method} method")
        
        mask_frame = mask_frame.astype(np.float32)
        live_frame = live_frame.astype(np.float32)
        
        vessel_info = {}
        
        if method == 'multi_criteria':
            vessel_mask = self._multi_criteria_vessel_detection(mask_frame, live_frame, vessel_info)
        elif method == 'statistical':
            vessel_mask = self._statistical_vessel_detection(mask_frame, live_frame, vessel_info)
        elif method == 'morphological':
            vessel_mask = self._morphological_vessel_detection(mask_frame, live_frame, vessel_info)
        else:
            raise ValueError(f"Unknown vessel detection method: {method}")
        
        vessel_mask = self._refine_vessel_mask(vessel_mask, vessel_info)
        self._analyze_vessel_characteristics(vessel_mask, live_frame, vessel_info)
        
        vessel_info['computation_time'] = time.time() - start_time
        vessel_info['vessel_pixel_ratio'] = np.sum(vessel_mask > 0) / vessel_mask.size
        
        logger.info(f"Vessel detection completed: {vessel_info['vessel_pixel_ratio']:.3f} ratio")
        
        return vessel_mask, vessel_info
    
    def _multi_criteria_vessel_detection(self, 
                                       mask_frame: np.ndarray, 
                                       live_frame: np.ndarray,
                                       vessel_info: Dict) -> np.ndarray:
        """Multi-criteria vessel detection"""
        
        # Criterion 1: Intensity-based
        mean_live = np.mean(live_frame)
        std_live = np.std(live_frame)
        intensity_threshold = mean_live + self.vessel_detection_sensitivity * std_live
        vessel_mask_intensity = (live_frame > intensity_threshold).astype(np.uint8) * 255
        
        # Criterion 2: Difference-based
        difference = np.abs(live_frame - mask_frame)
        mean_diff = np.mean(difference)
        std_diff = np.std(difference)
        diff_threshold = mean_diff + self.vessel_detection_sensitivity * std_diff
        vessel_mask_diff = (difference > diff_threshold).astype(np.uint8) * 255
        
        # Criterion 3: Gradient-based
        grad_x = cv2.Sobel(live_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(live_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_threshold = np.percentile(gradient_magnitude, 90)
        vessel_mask_grad = (gradient_magnitude > grad_threshold).astype(np.uint8) * 255
        
        # Criterion 4: Local contrast
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(live_frame, -1, kernel)
        local_contrast = live_frame - local_mean
        contrast_threshold = np.std(local_contrast) * 1.5
        vessel_mask_contrast = (local_contrast > contrast_threshold).astype(np.uint8) * 255
        
        # Combine criteria
        weights = [0.3, 0.3, 0.2, 0.2]
        combined_score = (weights[0] * vessel_mask_intensity.astype(np.float32) + 
                         weights[1] * vessel_mask_diff.astype(np.float32) + 
                         weights[2] * vessel_mask_grad.astype(np.float32) + 
                         weights[3] * vessel_mask_contrast.astype(np.float32)) / 255.0
        
        final_threshold = 0.4
        vessel_mask = (combined_score > final_threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'intensity_threshold': float(intensity_threshold),
            'diff_threshold': float(diff_threshold),
            'grad_threshold': float(grad_threshold),
            'contrast_threshold': float(contrast_threshold),
            'final_threshold': final_threshold
        })
        
        return vessel_mask
    
    def _statistical_vessel_detection(self, 
                                    mask_frame: np.ndarray, 
                                    live_frame: np.ndarray,
                                    vessel_info: Dict) -> np.ndarray:
        """Statistical hypothesis testing for vessel detection - optimized version"""
        
        # Use smaller window and stride for efficiency
        window_size = 11  # Reduced from 21
        stride = 5  # Process every 5th pixel for speed
        half_window = window_size // 2
        
        mask_padded = np.pad(mask_frame, half_window, mode='reflect')
        live_padded = np.pad(live_frame, half_window, mode='reflect')
        
        vessel_probability = np.zeros_like(mask_frame)
        
        # Use strided sampling for efficiency
        y_indices = range(half_window, mask_padded.shape[0] - half_window, stride)
        x_indices = range(half_window, mask_padded.shape[1] - half_window, stride)
        
        for i in y_indices:
            for j in x_indices:
                mask_window = mask_padded[i-half_window:i+half_window+1, 
                                        j-half_window:j+half_window+1]
                live_window = live_padded[i-half_window:i+half_window+1, 
                                        j-half_window:j+half_window+1]
                
                try:
                    diff = live_window.flatten() - mask_window.flatten()
                    if np.std(diff) > 1e-6:  # Only test if there's variation
                        statistic, p_value = stats.wilcoxon(diff, alternative='greater')
                        confidence = 1 - p_value if p_value < 0.05 else 0
                    else:
                        confidence = 0
                    vessel_probability[i-half_window, j-half_window] = confidence
                except (ValueError, Warning) as e:
                    logger.debug(f"Statistical test failed at ({i},{j}): {e}")
                    vessel_probability[i-half_window, j-half_window] = \
                        max(0, (np.mean(live_window) - np.mean(mask_window)) / (np.std(mask_window) + 1e-6))
        
        # Interpolate sparse results
        if stride > 1:
            from scipy.interpolate import griddata
            y_sparse, x_sparse = np.meshgrid(
                range(0, mask_frame.shape[0], stride),
                range(0, mask_frame.shape[1], stride),
                indexing='ij'
            )
            y_dense, x_dense = np.meshgrid(
                range(mask_frame.shape[0]),
                range(mask_frame.shape[1]),
                indexing='ij'
            )
            points = np.column_stack([y_sparse.ravel(), x_sparse.ravel()])
            values = vessel_probability[::stride, ::stride].ravel()
            vessel_probability = griddata(points, values, (y_dense, x_dense), method='linear', fill_value=0)
        
        probability_threshold = 0.7
        vessel_mask = (vessel_probability > probability_threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'statistical_method': 'wilcoxon_signed_rank',
            'probability_threshold': probability_threshold,
            'mean_probability': float(np.mean(vessel_probability)),
            'max_probability': float(np.max(vessel_probability))
        })
        
        return vessel_mask
    
    def _morphological_vessel_detection(self, 
                                      mask_frame: np.ndarray, 
                                      live_frame: np.ndarray,
                                      vessel_info: Dict) -> np.ndarray:
        """Morphology-based vessel detection"""
        
        vessel_responses = []
        scales = [3, 5, 7, 9, 11]
        
        for scale in scales:
            kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, max(1, scale//3)))
            kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
            
            tophat_line = cv2.morphologyEx(live_frame, cv2.MORPH_TOPHAT, kernel_line)
            tophat_circle = cv2.morphologyEx(live_frame, cv2.MORPH_TOPHAT, kernel_circle)
            
            vessel_response = np.maximum(tophat_line, tophat_circle)
            vessel_responses.append(vessel_response)
        
        combined_response = np.maximum.reduce(vessel_responses)
        
        threshold = np.mean(combined_response) + 2 * np.std(combined_response)
        vessel_mask = (combined_response > threshold).astype(np.uint8) * 255
        
        vessel_info.update({
            'morphological_scales': scales,
            'threshold': float(threshold),
            'max_response': float(np.max(combined_response))
        })
        
        return vessel_mask
    
    def _refine_vessel_mask(self, vessel_mask: np.ndarray, vessel_info: Dict) -> np.ndarray:
        """Refine vessel mask using morphological operations"""
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_open)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vessel_mask)
        
        min_area = 50
        max_aspect_ratio = 10
        
        refined_mask = np.zeros_like(vessel_mask)
        valid_components = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area >= min_area:
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if aspect_ratio <= max_aspect_ratio:
                    refined_mask[labels == i] = 255
                    valid_components += 1
        
        vessel_info.update({
            'total_components': num_labels - 1,
            'valid_components': valid_components
        })
        
        return refined_mask
    
    def _analyze_vessel_characteristics(self, 
                                      vessel_mask: np.ndarray, 
                                      live_frame: np.ndarray,
                                      vessel_info: Dict) -> None:
        """Analyze vessel characteristics"""
        
        if np.sum(vessel_mask) == 0:
            vessel_info['vessel_characteristics'] = {}
            return
        
        vessel_pixels = live_frame[vessel_mask > 0]
        
        characteristics = {
            'mean_intensity': float(np.mean(vessel_pixels)),
            'std_intensity': float(np.std(vessel_pixels)),
            'min_intensity': float(np.min(vessel_pixels)),
            'max_intensity': float(np.max(vessel_pixels)),
            'total_area': int(np.sum(vessel_mask > 0)),
            'intensity_range': float(np.max(vessel_pixels) - np.min(vessel_pixels))
        }
        
        # Compute skeleton safely
        try:
            skeleton = self.compute_skeleton(vessel_mask)
            characteristics['skeleton_length'] = int(np.sum(skeleton > 0))
            characteristics['tortuosity'] = float(characteristics['skeleton_length'] / (characteristics['total_area'] + 1e-6))
        except Exception as e:
            logger.warning(f"Skeleton computation failed: {e}")
            characteristics['skeleton_length'] = 0
            characteristics['tortuosity'] = 0.0
        
        vessel_info['vessel_characteristics'] = characteristics
    
    def create_registration_mask_advanced(self, 
                                        vessel_mask: np.ndarray,
                                        mask_frame: np.ndarray,
                                        method: str = 'adaptive') -> Tuple[np.ndarray, Dict]:
        """Advanced registration mask creation"""
        logger.info(f"Creating registration mask using {method} method")
        start_time = time.time()
        
        mask_info = {}
        
        if method == 'adaptive':
            registration_mask = self._adaptive_registration_mask(vessel_mask, mask_frame, mask_info)
        elif method == 'gradient_weighted':
            registration_mask = self._gradient_weighted_mask(vessel_mask, mask_frame, mask_info)
        else:
            registration_mask = cv2.bitwise_not(vessel_mask)
            
        mask_info['computation_time'] = time.time() - start_time
        mask_info['registration_pixel_ratio'] = float(np.sum(registration_mask > 0) / registration_mask.size)
        
        return registration_mask, mask_info
    
    def _adaptive_registration_mask(self, 
                                  vessel_mask: np.ndarray, 
                                  mask_frame: np.ndarray,
                                  mask_info: Dict) -> np.ndarray:
        """Create adaptive registration mask"""
        
        vessel_distance = cv2.distanceTransform(cv2.bitwise_not(vessel_mask), cv2.DIST_L2, 5)
        
        grad_x = cv2.Sobel(mask_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_reliability = gradient_magnitude / (np.max(gradient_magnitude) + 1e-6)
        
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(mask_frame, -1, kernel)
        local_variance = cv2.filter2D(mask_frame**2, -1, kernel) - local_mean**2
        variance_reliability = local_variance / (np.max(local_variance) + 1e-6)
        
        h, w = mask_frame.shape
        y, x = np.ogrid[:h, :w]
        border_distance = np.minimum(np.minimum(x, w-x), np.minimum(y, h-y))
        border_reliability = np.minimum(border_distance / 50.0, 1.0)
        
        combined_reliability = (0.4 * gradient_reliability + 
                              0.4 * variance_reliability + 
                              0.2 * border_reliability)
        
        spatial_weight = np.exp(-vessel_distance / 20.0)
        registration_weight = combined_reliability * spatial_weight
        
        weight_threshold = np.percentile(registration_weight, 30)
        registration_mask = (registration_weight > weight_threshold).astype(np.uint8) * 255
        registration_mask = cv2.bitwise_and(registration_mask, cv2.bitwise_not(vessel_mask))
        
        mask_info.update({
            'weight_threshold': float(weight_threshold),
            'mean_weight': float(np.mean(registration_weight))
        })
        
        return registration_mask
    
    def _gradient_weighted_mask(self, 
                               vessel_mask: np.ndarray, 
                               mask_frame: np.ndarray,
                               mask_info: Dict) -> np.ndarray:
        """Create gradient-weighted registration mask"""
        
        grad_x = cv2.Sobel(mask_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_threshold = np.percentile(gradient_magnitude, 50)
        registration_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255
        registration_mask = cv2.bitwise_and(registration_mask, cv2.bitwise_not(vessel_mask))
        
        mask_info['gradient_threshold'] = float(gradient_threshold)
        
        return registration_mask
    
    def estimate_motion_ecc_advanced(self, 
                                   mask_frame: np.ndarray, 
                                   live_frame: np.ndarray,
                                   registration_mask: Optional[np.ndarray] = None,
                                   initial_transform: Optional[np.ndarray] = None) -> MotionParameters:
        """Advanced ECC motion estimation"""
        logger.info("Advanced ECC motion estimation")
        start_time = time.time()
        
        if initial_transform is not None:
            transform_matrix = initial_transform.copy()
        else:
            transform_matrix = self.transform_matrix.copy()
        
        cv_motion_types = {
            MotionModel.TRANSLATION: cv2.MOTION_TRANSLATION,
            MotionModel.RIGID: cv2.MOTION_EUCLIDEAN,
            MotionModel.AFFINE: cv2.MOTION_AFFINE
        }
        
        if self.motion_model not in cv_motion_types:
            logger.warning(f"Motion model {self.motion_model} not supported by ECC, using AFFINE")
            motion_type = cv2.MOTION_AFFINE
        else:
            motion_type = cv_motion_types[self.motion_model]
        
        correlation_coeffs = []
        best_cc = -1
        best_transform = transform_matrix.copy()
        
        for level in range(self.multi_scale_levels, 0, -1):
            scale_factor = 2**(level-1)
            if scale_factor > 1:
                scaled_mask = cv2.resize(mask_frame, None, fx=1/scale_factor, fy=1/scale_factor)
                scaled_live = cv2.resize(live_frame, None, fx=1/scale_factor, fy=1/scale_factor)
                scaled_reg_mask = cv2.resize(registration_mask, None, 
                                           fx=1/scale_factor, fy=1/scale_factor) if registration_mask is not None else None
                scaled_transform = transform_matrix.copy()
                scaled_transform[:, 2] /= scale_factor
            else:
                scaled_mask = mask_frame
                scaled_live = live_frame
                scaled_reg_mask = registration_mask
                scaled_transform = transform_matrix
            
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                       self.max_iterations // self.multi_scale_levels, 
                       self.convergence_threshold)
            
            try:
                if scaled_reg_mask is not None:
                    cc, level_transform = cv2.findTransformECC(
                        scaled_mask, scaled_live, scaled_transform,
                        motion_type, criteria, scaled_reg_mask
                    )
                else:
                    cc, level_transform = cv2.findTransformECC(
                        scaled_mask, scaled_live, scaled_transform,
                        motion_type, criteria
                    )
                
                correlation_coeffs.append(cc)
                
                if scale_factor > 1:
                    level_transform[:, 2] *= scale_factor
                
                if cc > best_cc:
                    best_cc = cc
                    best_transform = level_transform.copy()
                
                transform_matrix = level_transform.copy()
                logger.info(f"Level {level}: CC = {cc:.6f}")
                
            except cv2.error as e:
                logger.warning(f"ECC failed at level {level}: {e}")
                correlation_coeffs.append(-1)
        
        parameters = self._extract_motion_parameters(best_transform)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        motion_params = MotionParameters(
            transform_matrix=best_transform,
            parameters=parameters,
            confidence=best_cc,
            convergence_iterations=len([cc for cc in correlation_coeffs if cc > 0]),
            correlation_coefficient=best_cc,
            inlier_ratio=1.0,
            computation_time=time.time() - start_time,
            method_used="ECC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"ECC completed: CC={best_cc:.6f}, Motion={motion_magnitude:.2f}px")
        
        return motion_params
    
    def estimate_motion_orb_ransac_advanced(self, 
                                          mask_frame: np.ndarray, 
                                          live_frame: np.ndarray,
                                          registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """Advanced ORB+RANSAC motion estimation"""
        logger.info("Advanced ORB+RANSAC motion estimation")
        start_time = time.time()
        
        kp1, des1 = self.orb_detector.detectAndCompute(mask_frame, registration_mask)
        kp2, des2 = self.orb_detector.detectAndCompute(live_frame, registration_mask)
        
        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            logger.warning("Insufficient features detected")
            return self._create_identity_motion_parameters("ORB_insufficient_features")
        
        matches = self._advanced_feature_matching(des1, des2, kp1, kp2)
        
        if len(matches) < 10:
            logger.warning("Insufficient good matches")
            return self._create_identity_motion_parameters("ORB_insufficient_matches")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        transform_matrix, inliers = self._advanced_ransac_estimation(src_pts, dst_pts)
        
        if transform_matrix is None:
            logger.warning("RANSAC estimation failed")
            return self._create_identity_motion_parameters("ORB_ransac_failed")
        
        validation_score = self._validate_transformation(src_pts, dst_pts, transform_matrix, inliers)
        
        parameters = self._extract_motion_parameters(transform_matrix)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        inlier_ratio = np.sum(inliers) / len(matches)
        
        motion_params = MotionParameters(
            transform_matrix=transform_matrix,
            parameters=parameters,
            confidence=validation_score,
            convergence_iterations=1,
            correlation_coefficient=validation_score,
            inlier_ratio=inlier_ratio,
            computation_time=time.time() - start_time,
            method_used="ORB_RANSAC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"ORB+RANSAC completed: Features={len(kp1)}/{len(kp2)}, Matches={len(matches)}, Inliers={np.sum(inliers)}")
        
        return motion_params
    
    def estimate_motion_sift_ransac_advanced(self,
                                           mask_frame: np.ndarray, 
                                           live_frame: np.ndarray,
                                           registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """SIFT-based motion estimation"""
        if self.sift_detector is None:
            logger.warning("SIFT not available, falling back to ORB")
            return self.estimate_motion_orb_ransac_advanced(mask_frame, live_frame, registration_mask)
        
        logger.info("SIFT+RANSAC motion estimation")
        start_time = time.time()
        
        kp1, des1 = self.sift_detector.detectAndCompute(mask_frame, registration_mask)
        kp2, des2 = self.sift_detector.detectAndCompute(live_frame, registration_mask)
        
        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            logger.warning("Insufficient SIFT features detected")
            return self._create_identity_motion_parameters("SIFT_insufficient_features")
        
        # FLANN matching for SIFT
        matches = self.flann_matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            logger.warning("Insufficient SIFT matches")
            return self._create_identity_motion_parameters("SIFT_insufficient_matches")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        transform_matrix, inliers = self._advanced_ransac_estimation(src_pts, dst_pts)
        
        if transform_matrix is None:
            logger.warning("SIFT RANSAC failed")
            return self._create_identity_motion_parameters("SIFT_ransac_failed")
        
        validation_score = self._validate_transformation(src_pts, dst_pts, transform_matrix, inliers)
        
        parameters = self._extract_motion_parameters(transform_matrix)
        motion_magnitude = np.linalg.norm(parameters[:2])
        angular_displacement = abs(parameters[2]) if len(parameters) > 2 else 0
        
        inlier_ratio = np.sum(inliers) / len(good_matches)
        
        motion_params = MotionParameters(
            transform_matrix=transform_matrix,
            parameters=parameters,
            confidence=validation_score,
            convergence_iterations=1,
            correlation_coefficient=validation_score,
            inlier_ratio=inlier_ratio,
            computation_time=time.time() - start_time,
            method_used="SIFT_RANSAC_advanced",
            motion_magnitude=motion_magnitude,
            angular_displacement=angular_displacement
        )
        
        logger.info(f"SIFT+RANSAC completed: Matches={len(good_matches)}, Inliers={np.sum(inliers)}")
        
        return motion_params
    
    def estimate_motion_phase_correlation(self,
                                        mask_frame: np.ndarray, 
                                        live_frame: np.ndarray) -> MotionParameters:
        """Phase correlation motion estimation for translation"""
        logger.info("Phase correlation motion estimation")
        start_time = time.time()
        
        # Phase correlation only works for translation
        try:
            # Convert to proper format
            mask_32f = mask_frame.astype(np.float32)
            live_32f = live_frame.astype(np.float32)
            
            # Compute phase correlation
            shift, response = cv2.phaseCorrelate(mask_32f, live_32f)
            
            tx, ty = shift
            
            # Create translation matrix
            if self.motion_model == MotionModel.TRANSLATION:
                transform_matrix = np.array([[1.0, 0.0, tx],
                                           [0.0, 1.0, ty]], dtype=np.float32)
            else:
                # Extend to requested model with translation only
                transform_matrix = self.transform_matrix.copy()
                transform_matrix[0, 2] = tx
                transform_matrix[1, 2] = ty
            
            parameters = self._extract_motion_parameters(transform_matrix)
            motion_magnitude = np.linalg.norm([tx, ty])
            
            # Response is the confidence measure
            confidence = float(response)
            
            motion_params = MotionParameters(
                transform_matrix=transform_matrix,
                parameters=parameters,
                confidence=confidence,
                convergence_iterations=1,
                correlation_coefficient=confidence,
                inlier_ratio=1.0,
                computation_time=time.time() - start_time,
                method_used="Phase_Correlation",
                motion_magnitude=motion_magnitude,
                angular_displacement=0.0
            )
            
            logger.info(f"Phase correlation completed: shift=({tx:.2f}, {ty:.2f}), response={confidence:.3f}")
            
            return motion_params
            
        except Exception as e:
            logger.error(f"Phase correlation failed: {e}")
            return self._create_identity_motion_parameters("Phase_Correlation_failed")
    
    def estimate_motion_hybrid(self,
                            mask_frame: np.ndarray, 
                            live_frame: np.ndarray,
                            registration_mask: Optional[np.ndarray] = None) -> MotionParameters:
        """Hybrid motion estimation combining multiple methods"""
        logger.info("Hybrid motion estimation - combining multiple methods")
        start_time = time.time()
        
        # Try multiple methods
        methods_results = []
        
        # Method 1: ECC
        try:
            ecc_params = self.estimate_motion_ecc_advanced(mask_frame, live_frame, registration_mask)
            methods_results.append(('ECC', ecc_params))
            logger.info(f"ECC confidence: {ecc_params.confidence:.3f}")
        except Exception as e:
            logger.warning(f"ECC failed: {e}")
        
        # Method 2: ORB
        try:
            orb_params = self.estimate_motion_orb_ransac_advanced(mask_frame, live_frame, registration_mask)
            methods_results.append(('ORB', orb_params))
            logger.info(f"ORB confidence: {orb_params.confidence:.3f}")
        except Exception as e:
            logger.warning(f"ORB failed: {e}")
        
        # Method 3: Phase Correlation (if translation model)
        if self.motion_model == MotionModel.TRANSLATION:
            try:
                phase_params = self.estimate_motion_phase_correlation(mask_frame, live_frame)
                methods_results.append(('Phase', phase_params))
                logger.info(f"Phase correlation confidence: {phase_params.confidence:.3f}")
            except Exception as e:
                logger.warning(f"Phase correlation failed: {e}")
        
        if not methods_results:
            logger.error("All hybrid methods failed")
            return self._create_identity_motion_parameters("Hybrid_all_failed")
        
        # Select best result based on confidence
        best_method, best_params = max(methods_results, key=lambda x: x[1].confidence)
        
        # Update method name
        best_params.method_used = f"Hybrid_{best_method}_selected"
        best_params.computation_time = time.time() - start_time
        
        logger.info(f"Hybrid selected: {best_method} with confidence {best_params.confidence:.3f}")
        
        return best_params
    
    def _advanced_feature_matching(self, 
                                 des1: np.ndarray, 
                                 des2: np.ndarray,
                                 kp1: List, 
                                 kp2: List) -> List:
        """Advanced feature matching with validation"""
        
        matches = self.bf_matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
                
            m, n = match_pair
            
            if m.distance < 0.75 * n.distance:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                
                displacement = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if displacement < 200:
                    scale1 = kp1[m.queryIdx].size
                    scale2 = kp2[m.trainIdx].size
                    scale_ratio = max(scale1, scale2) / (min(scale1, scale2) + 1e-6)
                    
                    if scale_ratio < 2.0:
                        good_matches.append(m)
        
        # Spatial clustering
        if len(good_matches) > 20:
            match_coords = []
            for m in good_matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                displacement_vec = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
                match_coords.append(displacement_vec)
            
            match_coords = np.array(match_coords)
            try:
                clustering = DBSCAN(eps=10, min_samples=3).fit(match_coords)
                
                if len(np.unique(clustering.labels_)) > 1:
                    valid_labels = clustering.labels_[clustering.labels_ >= 0]
                    if len(valid_labels) > 0:
                        largest_cluster = np.bincount(valid_labels).argmax()
                        cluster_mask = clustering.labels_ == largest_cluster
                        good_matches = [good_matches[i] for i in range(len(good_matches)) if cluster_mask[i]]
            except Exception as e:
                logger.debug(f"Clustering failed: {e}")
        
        return good_matches
    
    def _advanced_ransac_estimation(self, 
                                  src_pts: np.ndarray, 
                                  dst_pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advanced RANSAC estimation with adaptive parameters"""
        
        num_points = len(src_pts)
        confidence = 0.995
        
        initial_inlier_ratio = min(0.8, max(0.3, num_points / 100))
        
        min_samples = 3 if self.motion_model == MotionModel.RIGID else 4
        max_iterations = min(5000, int(np.log(1 - confidence) / 
                                    np.log(1 - initial_inlier_ratio**min_samples + 1e-10)))
        
        best_transform = None
        best_inliers = np.array([])
        max_inliers = 0
        
        thresholds = [self.outlier_threshold * 2, self.outlier_threshold, self.outlier_threshold * 0.5]
        
        for threshold in thresholds:
            if self.motion_model == MotionModel.TRANSLATION:
                transform, inliers = self._estimate_translation_ransac(
                    src_pts, dst_pts, threshold, max_iterations
                )
            elif self.motion_model == MotionModel.RIGID:
                transform, inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=threshold,
                    maxIters=max_iterations,
                    confidence=confidence
                )
            else:
                transform, inliers = cv2.estimateAffine2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=threshold,
                    maxIters=max_iterations,
                    confidence=confidence
                )
            
            if transform is not None and inliers is not None:
                num_inliers = np.sum(inliers)
                if num_inliers > max_inliers:
                    best_transform = transform
                    best_inliers = inliers.flatten()
                    max_inliers = num_inliers
        
        return best_transform, best_inliers
    
    def _estimate_translation_ransac(self, 
                                   src_pts: np.ndarray, 
                                   dst_pts: np.ndarray,
                                   threshold: float, 
                                   max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Custom RANSAC for translation-only model"""
        
        best_transform = None
        best_inliers = np.array([])
        max_inlier_count = 0
        
        num_pts = len(src_pts)
        if num_pts < 1:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), np.array([])
        
        for _ in range(max_iterations):
            sample_idx = np.random.choice(num_pts, 1, replace=False)
            
            translation = dst_pts[sample_idx] - src_pts[sample_idx]
            tx, ty = translation[0, 0]
            
            transform = np.array([[1.0, 0.0, tx],
                                [0.0, 1.0, ty]], dtype=np.float32)
            
            transformed_pts = src_pts + translation
            distances = np.linalg.norm(dst_pts - transformed_pts, axis=2).flatten()
            inliers = distances < threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_transform = transform
                best_inliers = inliers
        
        return best_transform, best_inliers
    
    def _validate_transformation(self, 
                               src_pts: np.ndarray, 
                               dst_pts: np.ndarray,
                               transform: np.ndarray, 
                               inliers: np.ndarray) -> float:
        """Statistical validation of estimated transformation"""
        
        if len(inliers) == 0 or np.sum(inliers) < 3:
            return 0.0
        
        src_homogeneous = np.hstack([src_pts.reshape(-1, 2), np.ones((len(src_pts), 1))])
        transformed_pts = (transform @ src_homogeneous.T).T
        
        inlier_mask = inliers.astype(bool)
        residuals = np.linalg.norm(dst_pts.reshape(-1, 2)[inlier_mask] - 
                                 transformed_pts[inlier_mask], axis=1)
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        inlier_ratio = np.sum(inliers) / len(inliers)
        residual_score = max(0, 1 - mean_residual / 10.0)
        stability_score = max(0, 1 - std_residual / 5.0)
        
        validation_score = (0.5 * inlier_ratio + 
                          0.3 * residual_score + 
                          0.2 * stability_score)
        
        return validation_score
    
    def _extract_motion_parameters(self, transform_matrix: np.ndarray) -> np.ndarray:
        """Extract motion parameters from transformation matrix"""
        
        if self.motion_model == MotionModel.TRANSLATION:
            return np.array([transform_matrix[0, 2], transform_matrix[1, 2]])
        
        elif self.motion_model == MotionModel.RIGID:
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            return np.array([tx, ty, angle])
        
        elif self.motion_model == MotionModel.SIMILARITY:
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            scale = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            return np.array([tx, ty, angle, scale])
        
        else:  # AFFINE
            return transform_matrix.flatten()[:6]
    
    def _create_identity_motion_parameters(self, method: str) -> MotionParameters:
        """Create identity motion parameters for failed cases"""
        return MotionParameters(
            transform_matrix=self.transform_matrix.copy(),
            parameters=np.zeros(self.dof),
            confidence=0.0,
            convergence_iterations=0,
            correlation_coefficient=0.0,
            inlier_ratio=0.0,
            computation_time=0.0,
            method_used=method,
            motion_magnitude=0.0,
            angular_displacement=0.0
        )
    
    def apply_transformation_advanced(self, 
                                    live_frame: np.ndarray,
                                    motion_params: MotionParameters,
                                    vessel_mask: Optional[np.ndarray] = None,
                                    preservation_method: str = 'selective') -> Tuple[np.ndarray, Dict]:
        """Advanced transformation application with vessel preservation"""
        logger.info(f"Applying transformation using {preservation_method} preservation")
        start_time = time.time()
        
        transform_info = {}
        
        if self.motion_model == MotionModel.PROJECTIVE:
            h, w = live_frame.shape[:2]
            aligned_frame = cv2.warpPerspective(live_frame, motion_params.transform_matrix, 
                                              (w, h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)
        else:
            aligned_frame = cv2.warpAffine(live_frame, motion_params.transform_matrix, 
                                         (live_frame.shape[1], live_frame.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
        
        if vessel_mask is not None and preservation_method != 'none':
            if preservation_method == 'selective':
                aligned_frame = self._selective_vessel_preservation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
            elif preservation_method == 'blended':
                aligned_frame = self._blended_vessel_preservation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
            elif preservation_method == 'intensity_preserved':
                aligned_frame = self._intensity_preserved_transformation(
                    live_frame, aligned_frame, vessel_mask, transform_info
                )
        
        transform_info['computation_time'] = time.time() - start_time
        transform_info['preservation_method'] = preservation_method
        
        return aligned_frame, transform_info
    
    def _selective_vessel_preservation(self, 
                                     original: np.ndarray,
                                     warped: np.ndarray, 
                                     vessel_mask: np.ndarray,
                                     transform_info: Dict) -> np.ndarray:
        """Selectively preserve vessel pixels"""
        
        result = warped.copy()
        vessel_pixels = vessel_mask > 0
        
        result[vessel_pixels] = original[vessel_pixels]
        
        transform_info['preserved_pixels'] = int(np.sum(vessel_pixels))
        transform_info['preservation_ratio'] = float(np.sum(vessel_pixels) / vessel_mask.size)
        
        return result
    
    def _blended_vessel_preservation(self, 
                                   original: np.ndarray,
                                   warped: np.ndarray, 
                                   vessel_mask: np.ndarray,
                                   transform_info: Dict) -> np.ndarray:
        """Smooth blending at vessel boundaries - FIXED"""
        
        vessel_distance = cv2.distanceTransform(cv2.bitwise_not(vessel_mask), cv2.DIST_L2, 5)
        
        blend_distance = 5.0
        weights = 1.0 / (1.0 + np.exp(-(vessel_distance - blend_distance) / 2.0))
        
        # Fixed dimension handling
        if len(original.shape) == 2:
            result = weights * warped + (1 - weights) * original
        else:
            result = weights[..., np.newaxis] * warped + (1 - weights[..., np.newaxis]) * original
        
        transform_info['blend_distance'] = blend_distance
        transform_info['blended_pixels'] = int(np.sum(weights < 0.99))
        
        return result.astype(original.dtype)
    
    def _intensity_preserved_transformation(self, 
                                          original: np.ndarray,
                                          warped: np.ndarray, 
                                          vessel_mask: np.ndarray,
                                          transform_info: Dict) -> np.ndarray:
        """Preserve vessel intensity relationships"""
        
        vessel_pixels = vessel_mask > 0
        
        if np.sum(vessel_pixels) == 0:
            return warped
        
        original_vessel_mean = np.mean(original[vessel_pixels])
        warped_vessel_mean = np.mean(warped[vessel_pixels])
        
        if warped_vessel_mean > 0:
            intensity_scale = original_vessel_mean / warped_vessel_mean
        else:
            intensity_scale = 1.0
        
        result = warped.copy()
        result[vessel_pixels] = np.clip(warped[vessel_pixels] * intensity_scale, 
                                       0, 255).astype(original.dtype)
        
        transform_info['intensity_scale_factor'] = float(intensity_scale)
        transform_info['corrected_pixels'] = int(np.sum(vessel_pixels))
        
        return result
    
    def compute_quality_metrics(self,