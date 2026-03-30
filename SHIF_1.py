import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import sys

# Fix for Windows DLL loading issue
if sys.platform == 'win32':
    try:
        import cupy as cp
        cupy_path = os.path.dirname(cp.__file__)
        
        # Add CuPy DLL paths
        dll_paths = [
            os.path.join(cupy_path, '.data'),
            os.path.join(cupy_path, 'cuda', 'bin'),
            os.path.join(cupy_path, '_core'),
        ]
        
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    os.add_dll_directory(dll_path)
                except:
                    pass
    except:
        pass

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy not found - using CPU only")
    print("  To enable GPU: pip install cupy-cuda12x (for CUDA 12.x)")
except Exception as e:
    CUPY_AVAILABLE = False
    print(f"⚠ CuPy import error: {e}")
    print("  Falling back to CPU processing")

# ============================================================
# GPU CONFIGURATION CHECK
# ============================================================

def check_gpu_support():
    """
    Check GPU availability
    """
    print("=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)
    
    # Check CUDA via CuPy
    if CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device(0)
            print(f"✓ CuPy GPU Available")
            print(f"✓ GPU Name: {device.attributes['Name'].decode()}")
            print(f"✓ Total Memory: {device.mem_info[1] / (1024**3):.2f} GB")
            print(f"✓ Free Memory: {device.mem_info[0] / (1024**3):.2f} GB")
            print(f"✓ Compute Capability: {device.compute_capability}")
            print(f"✓ Using GPU Device: 0")
            
            # Test simple operation
            test = cp.array([1, 2, 3])
            _ = test + 1
            print(f"✓ GPU operations working correctly")
            
            return True
        except Exception as e:
            print(f"⚠ GPU detected but error occurred: {e}")
            return False
    
    # Check OpenCV CUDA
    opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if opencv_cuda:
        print(f"✓ OpenCV CUDA Available")
        print(f"✓ CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.setDevice(0)
        return True
    
    print("⚠ No GPU acceleration available")
    print("  Running on CPU")
    print("\nTo enable GPU acceleration:")
    print("  Option 1: Install CuPy")
    print("    - For CUDA 12.x: pip install cupy-cuda12x")
    print("    - For CUDA 11.x: pip install cupy-cuda11x")
    print("  Option 2: Build OpenCV from source with CUDA")
    
    print("=" * 70)
    print()
    return False

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
            self.frames = np.expand_dims(pixel_array, axis=0)
        elif len(pixel_array.shape) == 3:
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
        """
        if self.frames is None:
            raise ValueError("Load DICOM first using load() method")
        
        print(f"Extracting frames...")
        print(f"  Mask frame: {mask_frame_idx}")
        print(f"  Live frames: {mask_frame_idx + 5} to {self.num_frames - 1}")
        
        self.mask_frame = self.frames[mask_frame_idx].copy()
        
        if mask_frame_idx == 0:
            self.live_frames = self.frames[1:].copy()
        else:
            self.live_frames = np.concatenate([
                self.frames[:mask_frame_idx],
                self.frames[mask_frame_idx + 1:]
            ], axis=0)
        
        print(f"✓ Extracted {len(self.live_frames)} live frames")
        print()
        
        return self.mask_frame, self.live_frames

# ============================================================
# GPU FRAME PROCESSOR (CuPy + OpenCV)
# ============================================================

class GPUFrameProcessor:
    """
    Process frames on GPU using CuPy
    """
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if self.use_gpu:
            print("GPU Processor initialized with CuPy")
        else:
            print("GPU Processor initialized with CPU fallback")
        
    def normalize_frame(self, frame):
        """Normalize frame to 0-255 range"""
        if self.use_gpu:
            gpu_frame = cp.asarray(frame)
            min_val = cp.min(gpu_frame)
            max_val = cp.max(gpu_frame)
            normalized = ((gpu_frame - min_val) / (max_val - min_val) * 255).astype(cp.uint8)
            return cp.asnumpy(normalized)
        else:
            return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def apply_clahe(self, frame, clip_limit=3.0, tile_grid_size=(8, 8)):
        """Apply CLAHE for contrast enhancement"""
        # CLAHE requires CPU (no GPU implementation in CuPy)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(frame)
        return enhanced
    
    def apply_bilateral_filter(self, frame, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter"""
        # Bilateral filter on CPU (complex for GPU)
        filtered = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        return filtered
    
    def apply_gaussian_blur(self, frame, ksize=(5, 5), sigma=2.0):
        """Apply Gaussian blur"""
        if self.use_gpu and ksize != (0, 0):
            gpu_frame = cp.asarray(frame, dtype=cp.float32)
            
            # Create Gaussian kernel
            kh, kw = ksize
            if kh % 2 == 0:
                kh += 1
            if kw % 2 == 0:
                kw += 1
            
            # Use CuPy's gaussian filter
            blurred = cp_ndimage.gaussian_filter(gpu_frame, sigma=sigma)
            return cp.asnumpy(blurred).astype(np.uint8)
        else:
            return cv2.GaussianBlur(frame, ksize, sigma)
    
    def unsharp_mask(self, frame, sigma=2.0, strength=1.5):
        """Apply unsharp masking"""
        blurred = self.apply_gaussian_blur(frame, ksize=(0, 0), sigma=sigma)
        
        if self.use_gpu:
            gpu_frame = cp.asarray(frame, dtype=cp.float32)
            gpu_blurred = cp.asarray(blurred, dtype=cp.float32)
            
            result = gpu_frame * (1.0 + strength) - gpu_blurred * strength
            result = cp.clip(result, 0, 255)
            
            return cp.asnumpy(result).astype(np.uint8)
        else:
            result = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def morphological_gradient(self, frame, kernel_size=3):
        """Apply morphological gradient"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
        return gradient
    
    def edge_enhancement(self, frame):
        """Enhanced edge detection using GPU"""
        if self.use_gpu:
            gpu_frame = cp.asarray(frame, dtype=cp.float32)
            
            # Sobel operators
            sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
            sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
            
            # Convolve
            from cupyx.scipy.signal import convolve2d
            grad_x = convolve2d(gpu_frame, sobel_x, mode='same', boundary='symm')
            grad_y = convolve2d(gpu_frame, sobel_y, mode='same', boundary='symm')
            
            # Magnitude
            magnitude = cp.sqrt(grad_x**2 + grad_y**2)
            magnitude = cp.clip(magnitude, 0, 255)
            
            return cp.asnumpy(magnitude).astype(np.uint8)
        else:
            grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)
    
    def preprocess_for_features(self, frame, aggressive=True):
        """
        Complete preprocessing pipeline for feature detection
        """
        # Step 1: Normalize to 8-bit
        if frame.dtype != np.uint8:
            frame = self.normalize_frame(frame)
        
        # Step 2: CLAHE
        clip_limit = 3.0 if aggressive else 2.0
        enhanced = self.apply_clahe(frame, clip_limit=clip_limit)
        
        # Step 3: Bilateral filter
        filtered = self.apply_bilateral_filter(enhanced, d=9, sigma_color=75, sigma_space=75)
        
        if aggressive:
            # Step 4: Unsharp masking
            sharpened = self.unsharp_mask(filtered, sigma=2.0, strength=1.5)
            
            # Step 5: Edge enhancement
            edges = self.edge_enhancement(filtered)
            
            # Step 6: Combine
            if self.use_gpu:
                gpu_sharp = cp.asarray(sharpened, dtype=cp.float32)
                gpu_edges = cp.asarray(edges, dtype=cp.float32)
                combined = gpu_sharp * 0.7 + gpu_edges * 0.3
                return cp.asnumpy(combined).astype(np.uint8)
            else:
                combined = cv2.addWeighted(sharpened, 0.7, edges, 0.3, 0)
                return combined.astype(np.uint8)
        else:
            return filtered
    
    def upload_to_gpu(self, mask_frame, live_frames):
        """
        Preprocess all frames
        """
        print("=" * 70)
        print("GPU PREPROCESSING")
        print("=" * 70)
        
        start_time = time.time()
        
        print("Processing mask frame...")
        processed_mask = self.preprocess_for_features(mask_frame, aggressive=True)
        
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
        print(f"  Mode: {'GPU (CuPy)' if self.use_gpu else 'CPU'}")
        print("=" * 70)
        print()
        
        return processed_mask, processed_live

# ============================================================
# FEATURE DETECTOR
# ============================================================

class DenseFeatureDetector:
    """
    Detect features with SIFT or dense sampling
    """
    def __init__(self, use_sift=True, dense_sampling=False, step_size=8):
        self.use_sift = use_sift
        self.dense_sampling = dense_sampling
        self.step_size = step_size
        
        if use_sift:
            self.detector = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=4,
                contrastThreshold=0.01,
                edgeThreshold=15,
                sigma=1.2
            )
        else:
            self.detector = cv2.ORB_create(
                nfeatures=10000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                patchSize=31
            )
    
    def create_dense_keypoints(self, frame_shape):
        """Create dense grid of keypoints"""
        h, w = frame_shape
        keypoints = []
        
        for y in range(0, h, self.step_size):
            for x in range(0, w, self.step_size):
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=float(self.step_size))
                keypoints.append(kp)
        
        return keypoints
    
    def detect_and_compute(self, frame, mask=None):
        """Detect features in frame"""
        if self.dense_sampling:
            keypoints = self.create_dense_keypoints(frame.shape)
            keypoints, descriptors = self.detector.compute(frame, keypoints)
        else:
            keypoints, descriptors = self.detector.detectAndCompute(frame, mask)
        
        return keypoints, descriptors
    
    def detect_features_in_frames(self, mask_frame, live_frames, roi_mask=None):
        """Detect features in mask and all live frames"""
        print("=" * 70)
        print("FEATURE DETECTION")
        print("=" * 70)
        print(f"Detector: {'SIFT' if self.use_sift else 'ORB'}")
        print(f"Mode: {'Dense sampling' if self.dense_sampling else 'Standard detection'}")
        if self.dense_sampling:
            print(f"Step size: {self.step_size} pixels")
            h, w = mask_frame.shape
            estimated = (h // self.step_size) * (w // self.step_size)
            print(f"Estimated features per frame: ~{estimated:,}")
        print()
        
        start_time = time.time()
        
        print("Detecting features in mask frame...")
        kp_mask, des_mask = self.detect_and_compute(mask_frame, roi_mask)
        print(f"  ✓ Detected {len(kp_mask) if kp_mask else 0:,} keypoints")
        
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
        avg_features = np.mean([len(kp) for kp, _ in live_features if kp is not None])
        print(f"  Average features per frame: {avg_features:,.0f}")
        print("=" * 70)
        print()
        
        return (kp_mask, des_mask), live_features

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    dicom_path = "D:/Rohith/Auto_pixel_shift/rohit test/1.2.826.0.1.3680043.2.1330.2640117.2411301755470003.5.3767_Raw.dcm"
    
    # Configuration
    USE_SIFT = True
    DENSE_SAMPLING = False
    STEP_SIZE = 8
    
    print("\n")
    print("=" * 70)
    print("DSA AUTOMATIC PIXEL SHIFT - GPU ACCELERATED PIPELINE")
    print("=" * 70)
    print()
    
    # Check GPU
    gpu_available = check_gpu_support()
    
    # Load DICOM
    reader = DicomFrameReader(dicom_path)
    reader.load()
    mask_frame, live_frames = reader.extract_mask_and_live_frames(mask_frame_idx=0)
    
    # Preprocess
    gpu_processor = GPUFrameProcessor(use_gpu=gpu_available)
    processed_mask, processed_live = gpu_processor.upload_to_gpu(mask_frame, live_frames)
    
    # Feature detection
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
    
    # Visualize
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    kp_mask, des_mask = mask_features
    kp_live_0, des_live_0 = live_features[0]
    
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
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(mask_frame, cmap='gray')
    axes[0, 0].set_title(f'Original Mask Frame')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(live_frames[0], cmap='gray')
    axes[0, 1].set_title(f'Original Live Frame 1')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(live_frames[0] - mask_frame), cmap='hot')
    axes[0, 2].set_title('Difference (Raw)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(mask_vis)
    axes[1, 0].set_title(f'Mask: {len(kp_mask):,} Features')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(live_vis)
    axes[1, 1].set_title(f'Live: {len(kp_live_0):,} Features')
    axes[1, 1].axis('off')
    
    if len(kp_mask) > 0:
        h, w = processed_mask.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        for kp in kp_mask:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap, (x, y), 10, 1, -1)
        
        axes[1, 2].imshow(heatmap, cmap='hot')
        axes[1, 2].set_title('Feature Heatmap')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gpu_feature_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"GPU Acceleration: {'CuPy' if gpu_available else 'CPU only'}")
    print(f"Frames processed: {len(live_frames)}")
    print(f"Mask features: {len(kp_mask):,}")
    print(f"Live frame 1 features: {len(kp_live_0):,}")
    print("=" * 70)