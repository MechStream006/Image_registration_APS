import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ============================================================
# GPU CONFIGURATION CHECK
# ============================================================

def check_gpu_support():
    """
    Check if OpenCV is built with CUDA support
    """
    print("=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)
    
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    if cuda_available:
        print(f"✓ CUDA Devices Available: {cv2.cuda.getCudaEnabledDeviceCount()}")
        
        # Get device info
        device_info = cv2.cuda.DeviceInfo()
        print(f"✓ GPU Name: {device_info.name()}")
        print(f"✓ Total Memory: {device_info.totalMemory() / (1024**3):.2f} GB")
        print(f"✓ Free Memory: {device_info.freeMemory() / (1024**3):.2f} GB")
        print(f"✓ Compute Capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
        
        # Set device
        cv2.cuda.setDevice(0)
        print(f"✓ Using GPU Device: 0")
    else:
        print("⚠ WARNING: No CUDA-enabled GPU detected!")
        print("  OpenCV will fall back to CPU processing")
        print("  To enable GPU:")
        print("  1. Install CUDA Toolkit")
        print("  2. Install opencv-contrib-python compiled with CUDA")
        print("     pip install opencv-contrib-python-headless")
    
    print("=" * 70)
    print()
    
    return cuda_available

# ============================================================
# DICOM READING AND FRAME EXTRACTION
# ============================================================

class DicomFrameReader:
    """
    Read DICOM cine loop and extract frames
    """
    def __init__(self, dicom_path):
        self.dicom_path = Path(dicom_path)
        self.ds = None
        self.frames = None
        self.mask_frame = None
        self.live_frames = None
        self.num_frames = 0
        self.frame_shape = None
        
    def load(self):
        """Load DICOM file and extract frames"""
        print("Loading DICOM file...")
        print(f"Path: {self.dicom_path}")
        
        # Read DICOM
        self.ds = pydicom.dcmread(str(self.dicom_path))
        
        # Extract pixel array
        pixel_array = self.ds.pixel_array
        
        # Handle different DICOM structures
        if len(pixel_array.shape) == 2:
            # Single frame
            self.frames = np.expand_dims(pixel_array, axis=0)
        elif len(pixel_array.shape) == 3:
            # Multi-frame
            self.frames = pixel_array
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")
        
        # Convert to float32 for processing
        self.frames = self.frames.astype(np.float32)
        
        # Get dimensions
        self.num_frames = self.frames.shape[0]
        self.frame_shape = (self.frames.shape[1], self.frames.shape[2])
        
        print(f"✓ Loaded successfully")
        print(f"  Total frames: {self.num_frames}")
        print(f"  Frame dimensions: {self.frame_shape[0]} × {self.frame_shape[1]}")
        print(f"  Bit depth: {self.ds.BitsAllocated if hasattr(self.ds, 'BitsAllocated') else 'Unknown'}")
        print(f"  Pixel value range: [{self.frames.min():.0f}, {self.frames.max():.0f}]")
        print()
        
        return self
    
    def extract_mask_and_live_frames(self, mask_frame_idx=0):
        """
        Extract mask frame and live frames
        
        Args:
            mask_frame_idx: Index of mask frame (default: 0)
        """
        if self.frames is None:
            raise ValueError("Load DICOM first using load() method")
        
        print(f"Extracting frames...")
        print(f"  Mask frame: {mask_frame_idx}")
        print(f"  Live frames: {mask_frame_idx + 1} to {self.num_frames - 1}")
        
        # Extract mask frame
        self.mask_frame = self.frames[mask_frame_idx].copy()
        
        # Extract live frames (all except mask)
        if mask_frame_idx == 0:
            self.live_frames = self.frames[1:].copy()
        else:
            # Handle case where mask is not first frame
            self.live_frames = np.concatenate([
                self.frames[:mask_frame_idx],
                self.frames[mask_frame_idx + 1:]
            ], axis=0)
        
        print(f"✓ Extracted {len(self.live_frames)} live frames")
        print()
        
        return self.mask_frame, self.live_frames
    
    def get_frame_info(self):
        """Get frame information"""
        return {
            'num_frames': self.num_frames,
            'frame_shape': self.frame_shape,
            'num_live_frames': len(self.live_frames) if self.live_frames is not None else 0,
            'pixel_spacing': self.ds.PixelSpacing if hasattr(self.ds, 'PixelSpacing') else None,
            'series_description': self.ds.SeriesDescription if hasattr(self.ds, 'SeriesDescription') else None
        }

# ============================================================
# GPU FRAME PROCESSOR
# ============================================================

class GPUFrameProcessor:
    """
    Process frames on GPU for feature detection preprocessing
    """
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and check_gpu_support()
        self.gpu_mask = None
        self.gpu_live_frames = []
        
    def normalize_frame(self, frame):
        """
        Normalize frame to 0-255 range
        """
        if self.use_gpu:
            # GPU version
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Normalize
            gpu_normalized = cv2.cuda.normalize(
                gpu_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            
            return gpu_normalized.download()
        else:
            # CPU version
            normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return normalized
    
    def apply_clahe(self, frame, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE for contrast enhancement
        """
        if self.use_gpu:
            # GPU CLAHE
            clahe = cv2.cuda.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size
            )
            
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            gpu_enhanced = clahe.apply(gpu_frame, cv2.cuda_Stream.Null())
            
            return gpu_enhanced.download()
        else:
            # CPU CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(frame)
            return enhanced
    
    def apply_bilateral_filter(self, frame, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for edge-preserving smoothing
        """
        if self.use_gpu:
            # GPU bilateral filter
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            gpu_filtered = cv2.cuda.bilateralFilter(
                gpu_frame, d, sigma_color, sigma_space
            )
            
            return gpu_filtered.download()
        else:
            # CPU bilateral filter
            filtered = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
            return filtered
    
    def apply_gaussian_blur(self, frame, ksize=(5, 5), sigma=2.0):
        """
        Apply Gaussian blur
        """
        if self.use_gpu:
            # GPU Gaussian blur
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            gaussian_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, ksize, sigma
            )
            
            gpu_blurred = gaussian_filter.apply(gpu_frame)
            
            return gpu_blurred.download()
        else:
            # CPU Gaussian blur
            blurred = cv2.GaussianBlur(frame, ksize, sigma)
            return blurred
    
    def unsharp_mask(self, frame, sigma=2.0, strength=1.5):
        """
        Apply unsharp masking for edge enhancement
        """
        # Blur
        blurred = self.apply_gaussian_blur(frame, ksize=(0, 0), sigma=sigma)
        
        # Unsharp mask
        if self.use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_blurred = cv2.cuda_GpuMat()
            gpu_frame.upload(frame.astype(np.float32))
            gpu_blurred.upload(blurred.astype(np.float32))
            
            # result = frame * (1 + strength) - blurred * strength
            gpu_result = cv2.cuda.addWeighted(
                gpu_frame, 1.0 + strength,
                gpu_blurred, -strength,
                0
            )
            
            result = gpu_result.download()
        else:
            result = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def morphological_gradient(self, frame, kernel_size=3):
        """
        Apply morphological gradient for edge detection
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if self.use_gpu:
            # GPU morphology
            morph_filter = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_GRADIENT, cv2.CV_8U, kernel
            )
            
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            gpu_gradient = morph_filter.apply(gpu_frame)
            
            return gpu_gradient.download()
        else:
            # CPU morphology
            gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
            return gradient
    
    def preprocess_for_features(self, frame, aggressive=True):
        """
        Complete preprocessing pipeline for feature detection
        
        Args:
            frame: Input frame (float32 or uint8)
            aggressive: If True, apply aggressive enhancement
        
        Returns:
            Enhanced frame optimized for feature detection
        """
        # Step 1: Normalize to 8-bit
        if frame.dtype != np.uint8:
            frame = self.normalize_frame(frame)
        
        # Step 2: CLAHE
        clip_limit = 3.0 if aggressive else 2.0
        enhanced = self.apply_clahe(frame, clip_limit=clip_limit)
        
        # Step 3: Bilateral filter (preserve edges, reduce noise)
        filtered = self.apply_bilateral_filter(enhanced, d=9, sigma_color=75, sigma_space=75)
        
        if aggressive:
            # Step 4: Unsharp masking
            sharpened = self.unsharp_mask(filtered, sigma=2.0, strength=1.5)
            
            # Step 5: Morphological gradient
            gradient = self.morphological_gradient(filtered, kernel_size=3)
            
            # Step 6: Combine original with gradient
            if self.use_gpu:
                gpu_sharp = cv2.cuda_GpuMat()
                gpu_grad = cv2.cuda_GpuMat()
                gpu_sharp.upload(sharpened.astype(np.float32))
                gpu_grad.upload(gradient.astype(np.float32))
                
                gpu_combined = cv2.cuda.addWeighted(gpu_sharp, 0.7, gpu_grad, 0.3, 0)
                combined = gpu_combined.download().astype(np.uint8)
            else:
                combined = cv2.addWeighted(sharpened, 0.7, gradient, 0.3, 0).astype(np.uint8)
            
            return combined
        else:
            return filtered
    
    def upload_to_gpu(self, mask_frame, live_frames):
        """
        Upload mask and live frames to GPU and preprocess
        
        Returns:
            processed_mask, list of processed_live_frames
        """
        print("=" * 70)
        print("GPU PREPROCESSING")
        print("=" * 70)
        
        start_time = time.time()
        
        # Preprocess mask frame
        print("Processing mask frame...")
        processed_mask = self.preprocess_for_features(mask_frame, aggressive=True)
        
        # Preprocess live frames
        print(f"Processing {len(live_frames)} live frames...")
        processed_live = []
        
        for i, frame in enumerate(live_frames):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Frame {i + 1}/{len(live_frames)}...", end="\r")
            
            processed = self.preprocess_for_features(frame, aggressive=True)
            processed_live.append(processed)
        
        print(f"\n✓ Preprocessing complete")
        
        elapsed = time.time() - start_time
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Average per frame: {elapsed / (len(live_frames) + 1):.3f} seconds")
        print(f"  Mode: {'GPU' if self.use_gpu else 'CPU'}")
        print("=" * 70)
        print()
        
        return processed_mask, processed_live

# ============================================================
# FEATURE DETECTOR (DENSE FEATURES)
# ============================================================

class DenseFeatureDetector:
    """
    Detect features densely across the entire frame
    For 1344x1344 images, we can detect features at every N pixels
    """
    def __init__(self, use_sift=True, dense_sampling=False, step_size=8):
        """
        Args:
            use_sift: Use SIFT detector (True) or ORB (False)
            dense_sampling: If True, sample features on a dense grid
            step_size: Pixel spacing for dense sampling
        """
        self.use_sift = use_sift
        self.dense_sampling = dense_sampling
        self.step_size = step_size
        
        if use_sift:
            # SIFT with aggressive parameters for DSA
            self.detector = cv2.SIFT_create(
                nfeatures=0,  # Detect as many as possible
                nOctaveLayers=4,
                contrastThreshold=0.01,  # Very low for DSA
                edgeThreshold=15,
                sigma=1.2
            )
        else:
            # ORB with many features
            self.detector = cv2.ORB_create(
                nfeatures=10000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                patchSize=31
            )
    
    def create_dense_keypoints(self, frame_shape):
        """
        Create dense grid of keypoints
        
        For 1344x1344, with step_size=8, this gives ~28,000 keypoints
        """
        h, w = frame_shape
        keypoints = []
        
        for y in range(0, h, self.step_size):
            for x in range(0, w, self.step_size):
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=self.step_size)
                keypoints.append(kp)
        
        return keypoints
    
    def detect_and_compute(self, frame, mask=None):
        """
        Detect features in frame
        
        Args:
            frame: Preprocessed frame
            mask: Optional ROI mask
        
        Returns:
            keypoints, descriptors
        """
        if self.dense_sampling:
            # Create dense keypoints
            keypoints = self.create_dense_keypoints(frame.shape)
            
            # Compute descriptors at dense locations
            keypoints, descriptors = self.detector.compute(frame, keypoints)
        else:
            # Standard feature detection
            keypoints, descriptors = self.detector.detectAndCompute(frame, mask)
        
        return keypoints, descriptors
    
    def detect_features_in_frames(self, mask_frame, live_frames, roi_mask=None):
        """
        Detect features in mask and all live frames
        
        Returns:
            mask_features: (keypoints, descriptors)
            live_features: list of (keypoints, descriptors) for each live frame
        """
        print("=" * 70)
        print("FEATURE DETECTION")
        print("=" * 70)
        print(f"Detector: {'SIFT' if self.use_sift else 'ORB'}")
        print(f"Mode: {'Dense sampling' if self.dense_sampling else 'Standard detection'}")
        if self.dense_sampling:
            print(f"Step size: {self.step_size} pixels")
            h, w = mask_frame.shape
            estimated_features = (h // self.step_size) * (w // self.step_size)
            print(f"Estimated features: ~{estimated_features:,}")
        print()
        
        start_time = time.time()
        
        # Detect in mask frame
        print("Detecting features in mask frame...")
        kp_mask, des_mask = self.detect_and_compute(mask_frame, roi_mask)
        print(f"  ✓ Detected {len(kp_mask) if kp_mask else 0:,} keypoints")
        
        # Detect in live frames
        print(f"Detecting features in {len(live_frames)} live frames...")
        live_features = []
        
        for i, frame in enumerate(live_frames):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Frame {i + 1}/{len(live_frames)}...", end="\r")
            
            kp, des = self.detect_and_compute(frame, roi_mask)
            live_features.append((kp, des))
        
        print(f"\n✓ Feature detection complete")
        
        elapsed = time.time() - start_time
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Average per frame: {elapsed / (len(live_frames) + 1):.3f} seconds")
        print("=" * 70)
        print()
        
        return (kp_mask, des_mask), live_features

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Configuration
    dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
    
    # Feature detection parameters
    USE_SIFT = True          # True for SIFT, False for ORB
    DENSE_SAMPLING = False   # True for dense grid sampling
    STEP_SIZE = 8           # Pixel spacing for dense sampling
    AGGRESSIVE_PREPROCESS = True
    
    print("\n")
    print("=" * 70)
    print("DSA AUTOMATIC PIXEL SHIFT - GPU ACCELERATED PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Read DICOM and extract frames
    reader = DicomFrameReader(dicom_path)
    reader.load()
    mask_frame, live_frames = reader.extract_mask_and_live_frames(mask_frame_idx=0)
    
    # Step 2: Upload to GPU and preprocess
    gpu_processor = GPUFrameProcessor(use_gpu=True)
    processed_mask, processed_live = gpu_processor.upload_to_gpu(mask_frame, live_frames)
    
    # Step 3: Feature detection
    detector = DenseFeatureDetector(
        use_sift=USE_SIFT,
        dense_sampling=DENSE_SAMPLING,
        step_size=STEP_SIZE
    )
    
    mask_features, live_features = detector.detect_features_in_frames(
        processed_mask, 
        processed_live,
        roi_mask=None
    )
    
    # Step 4: Visualize results for first live frame
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    kp_mask, des_mask = mask_features
    kp_live_0, des_live_0 = live_features[0]
    
    # Draw keypoints
    mask_vis = cv2.drawKeypoints(
        processed_mask, kp_mask, None, 
        color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    live_vis = cv2.drawKeypoints(
        processed_live[0], kp_live_0, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original frames
    axes[0, 0].imshow(mask_frame, cmap='gray')
    axes[0, 0].set_title(f'Original Mask Frame\nRange: [{mask_frame.min():.0f}, {mask_frame.max():.0f}]')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(live_frames[0], cmap='gray')
    axes[0, 1].set_title(f'Original Live Frame 1\nRange: [{live_frames[0].min():.0f}, {live_frames[0].max():.0f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(live_frames[0] - mask_frame), cmap='hot')
    axes[0, 2].set_title('Difference (Before Processing)')
    axes[0, 2].axis('off')
    
    # Row 2: Processed frames with features
    axes[1, 0].imshow(mask_vis)
    axes[1, 0].set_title(f'Mask: {len(kp_mask):,} Features Detected')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(live_vis)
    axes[1, 1].set_title(f'Live Frame 1: {len(kp_live_0):,} Features Detected')
    axes[1, 1].axis('off')
    
    # Feature distribution heatmap
    if len(kp_mask) > 0:
        h, w = processed_mask.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        for kp in kp_mask:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap, (x, y), 10, 1, -1)
        
        axes[1, 2].imshow(heatmap, cmap='hot')
        axes[1, 2].set_title('Feature Distribution Heatmap')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gpu_feature_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n")
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"DICOM file: {Path(dicom_path).name}")
    print(f"Total frames: {reader.num_frames}")
    print(f"Frame dimensions: {reader.frame_shape[0]} × {reader.frame_shape[1]}")
    print(f"Live frames processed: {len(live_frames)}")
    print()
    print(f"Feature Detector: {'SIFT' if USE_SIFT else 'ORB'}")
    print(f"Dense sampling: {DENSE_SAMPLING}")
    print(f"Mask features: {len(kp_mask):,}")
    print(f"Live frame 1 features: {len(kp_live_0):,}")
    print()
    print(f"GPU acceleration: {gpu_processor.use_gpu}")
    if des_mask is not None:
        print(f"Descriptor size: {des_mask.shape[1]} dimensions")
    print("=" * 70)
    
    # Save feature data
    print("\nFeature data available in variables:")
    print("  - processed_mask: Enhanced mask frame")
    print("  - processed_live: List of enhanced live frames")
    print("  - mask_features: (keypoints, descriptors) for mask")
    print("  - live_features: List of (keypoints, descriptors) for each live frame")
    print()
    print("Ready for next step: Feature matching and motion estimation!")
    print("=" * 70)