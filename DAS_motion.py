"""
DSA Motion Correction Visualization and Analysis Tools
Provides comprehensive visualization of registration quality
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Tuple
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


class DSAVisualizer:
    """Comprehensive visualization tools for DSA processing results"""
    
    def __init__(self, results: Dict):
        self.results = results
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_comprehensive_report(self, output_path: Path):
        """Generate complete visual report"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Registration quality overview
        self.plot_registration_quality(output_path / "registration_quality.png")
        
        # 2. Before/After comparison
        self.plot_before_after_comparison(output_path / "before_after_comparison.png")
        
        # 3. DSA quality assessment
        self.plot_dsa_quality(output_path / "dsa_quality.png")
        
        # 4. Motion analysis
        self.plot_motion_analysis(output_path / "motion_analysis.png")
        
        # 5. Frame-by-frame metrics
        self.plot_frame_metrics(output_path / "frame_metrics.png")
        
        # 6. Difference maps
        self.create_difference_maps(output_path / "difference_maps.png")
        
        # 7. Create animated GIF comparisons
        self.create_animations(output_path)
        
        print(f"Comprehensive report saved to: {output_path}")
    
    def plot_registration_quality(self, output_file: Path):
        """Plot overall registration quality metrics"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        metrics = self.results['quality_metrics']
        per_frame = metrics['per_frame']
        
        frame_indices = [m['frame_idx'] for m in per_frame]
        mi_values = [m['mutual_information'] for m in per_frame]
        ssim_values = [m['ssim'] for m in per_frame]
        sharpness_values = [m['edge_sharpness'] for m in per_frame]
        
        # Mutual Information plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(frame_indices, mi_values, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=metrics['overall']['mean_mi'], color='r', linestyle='--', 
                   label=f"Mean: {metrics['overall']['mean_mi']:.3f}")
        ax1.fill_between(frame_indices, 
                        metrics['overall']['mean_mi'] - metrics['overall']['std_mi'],
                        metrics['overall']['mean_mi'] + metrics['overall']['std_mi'],
                        alpha=0.2, color='red')
        ax1.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mutual Information', fontsize=12, fontweight='bold')
        ax1.set_title('Mutual Information (Higher = Better Alignment)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SSIM plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(frame_indices, ssim_values, 's-', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=metrics['overall']['mean_ssim'], color='r', linestyle='--',
                   label=f"Mean: {metrics['overall']['mean_ssim']:.3f}")
        ax2.fill_between(frame_indices,
                        metrics['overall']['mean_ssim'] - metrics['overall']['std_ssim'],
                        metrics['overall']['mean_ssim'] + metrics['overall']['std_ssim'],
                        alpha=0.2, color='red')
        ax2.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.set_title('Structural Similarity Index (0-1, Higher = Better)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Edge Sharpness plot
        ax3 = fig.add_subplot(gs[2, :2])
        ax3.plot(frame_indices, sharpness_values, '^-', linewidth=2, markersize=8, color='purple')
        ax3.axhline(y=metrics['overall']['mean_sharpness'], color='r', linestyle='--',
                   label=f"Mean: {metrics['overall']['mean_sharpness']:.2f}")
        ax3.fill_between(frame_indices,
                        metrics['overall']['mean_sharpness'] - metrics['overall']['std_sharpness'],
                        metrics['overall']['mean_sharpness'] + metrics['overall']['std_sharpness'],
                        alpha=0.2, color='red')
        ax3.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Edge Sharpness (Laplacian Variance)', fontsize=12, fontweight='bold')
        ax3.set_title('Edge Sharpness (Higher = Sharper Vessels)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics box
        ax4 = fig.add_subplot(gs[:, 2])
        ax4.axis('off')
        
        summary_text = "REGISTRATION QUALITY SUMMARY\n" + "="*40 + "\n\n"
        summary_text += f"Total Frames: {len(self.results['registered_frames'])}\n\n"
        summary_text += "MUTUAL INFORMATION\n"
        summary_text += f"  Mean: {metrics['overall']['mean_mi']:.4f}\n"
        summary_text += f"  Std:  {metrics['overall']['std_mi']:.4f}\n"
        summary_text += f"  Min:  {min(mi_values):.4f}\n"
        summary_text += f"  Max:  {max(mi_values):.4f}\n\n"
        
        summary_text += "STRUCTURAL SIMILARITY\n"
        summary_text += f"  Mean: {metrics['overall']['mean_ssim']:.4f}\n"
        summary_text += f"  Std:  {metrics['overall']['std_ssim']:.4f}\n"
        summary_text += f"  Min:  {min(ssim_values):.4f}\n"
        summary_text += f"  Max:  {max(ssim_values):.4f}\n\n"
        
        summary_text += "EDGE SHARPNESS\n"
        summary_text += f"  Mean: {metrics['overall']['mean_sharpness']:.2f}\n"
        summary_text += f"  Std:  {metrics['overall']['std_sharpness']:.2f}\n"
        summary_text += f"  Min:  {min(sharpness_values):.2f}\n"
        summary_text += f"  Max:  {max(sharpness_values):.2f}\n\n"
        
        # Quality assessment
        avg_ssim = metrics['overall']['mean_ssim']
        if avg_ssim > 0.95:
            quality = "EXCELLENT"
            color = 'green'
        elif avg_ssim > 0.90:
            quality = "GOOD"
            color = 'blue'
        elif avg_ssim > 0.85:
            quality = "ACCEPTABLE"
            color = 'orange'
        else:
            quality = "POOR - Review Recommended"
            color = 'red'
        
        summary_text += f"OVERALL QUALITY: {quality}"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        plt.suptitle('DSA Motion Correction - Registration Quality Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_before_after_comparison(self, output_file: Path):
        """Compare original vs registered frames"""
        fig = plt.figure(figsize=(18, 12))
        
        # Select key frames to display (first, middle, last contrast frames)
        mask_idx = self.results['mask_frame_idx']
        n_frames = len(self.results['raw_frames'])
        
        # Select 3 representative frames (excluding mask)
        frame_indices = []
        for i in range(n_frames):
            if i != mask_idx:
                frame_indices.append(i)
        
        if len(frame_indices) >= 3:
            display_indices = [
                frame_indices[0],  # First contrast
                frame_indices[len(frame_indices)//2],  # Middle
                frame_indices[-1]  # Last contrast
            ]
        else:
            display_indices = frame_indices[:3]
        
        for idx, frame_num in enumerate(display_indices):
            # Original preprocessed
            ax1 = plt.subplot(3, 4, idx*4 + 1)
            ax1.imshow(self.results['preprocessed_frames'][frame_num], cmap='gray')
            ax1.set_title(f'Original Frame {frame_num}', fontweight='bold')
            ax1.axis('off')
            
            # Registered
            ax2 = plt.subplot(3, 4, idx*4 + 2)
            ax2.imshow(self.results['registered_frames'][frame_num], cmap='gray')
            ax2.set_title(f'Registered Frame {frame_num}', fontweight='bold')
            ax2.axis('off')
            
            # Difference map (before registration)
            mask_frame = self.results['preprocessed_frames'][mask_idx]
            diff_before = cv2.absdiff(
                self.results['preprocessed_frames'][frame_num],
                mask_frame
            )
            ax3 = plt.subplot(3, 4, idx*4 + 3)
            im3 = ax3.imshow(diff_before, cmap='hot')
            ax3.set_title('Diff Before (Hot)', fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # Difference map (after registration)
            mask_registered = self.results['registered_frames'][mask_idx]
            diff_after = cv2.absdiff(
                self.results['registered_frames'][frame_num],
                mask_registered
            )
            ax4 = plt.subplot(3, 4, idx*4 + 4)
            im4 = ax4.imshow(diff_after, cmap='hot')
            ax4.set_title('Diff After (Hot)', fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        plt.suptitle('Before vs After Registration Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dsa_quality(self, output_file: Path):
        """Visualize DSA image quality"""
        fig = plt.figure(figsize=(20, 12))
        
        dsa_images = self.results['dsa_images']
        mask_idx = self.results['mask_frame_idx']
        
        # Select frames to display
        display_indices = []
        for i in range(len(dsa_images)):
            if i != mask_idx:
                display_indices.append(i)
        
        # Display up to 12 frames
        n_display = min(12, len(display_indices))
        n_cols = 4
        n_rows = (n_display + n_cols - 1) // n_cols
        
        for idx in range(n_display):
            frame_idx = display_indices[idx]
            dsa_img = dsa_images[frame_idx]
            
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Apply colormap for better visualization
            im = ax.imshow(dsa_img, cmap='hot', vmin=0, vmax=np.percentile(dsa_img, 99))
            ax.set_title(f'DSA Frame {frame_idx}', fontweight='bold', fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        plt.suptitle('DSA Images - Vessel Visualization', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_motion_analysis(self, output_file: Path):
        """Analyze and visualize motion patterns"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        matrices = self.results['transformation_matrices']
        mask_idx = self.results['mask_frame_idx']
        
        # Extract motion parameters
        translations_x = []
        translations_y = []
        rotations = []
        scales = []
        
        for i, matrix in enumerate(matrices):
            if i == mask_idx:
                continue
            
            # Extract translation
            tx = matrix[0, 2]
            ty = matrix[1, 2]
            translations_x.append(tx)
            translations_y.append(ty)
            
            # Extract rotation (approximate)
            rotation = np.arctan2(matrix[1, 0], matrix[0, 0]) * 180 / np.pi
            rotations.append(rotation)
            
            # Extract scale (approximate)
            scale_x = np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
            scale_y = np.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2)
            scales.append((scale_x + scale_y) / 2)
        
        frame_nums = [i for i in range(len(matrices)) if i != mask_idx]
        
        # Translation X
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(frame_nums, translations_x, 'o-', linewidth=2, markersize=6)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Frame Index', fontweight='bold')
        ax1.set_ylabel('Translation X (pixels)', fontweight='bold')
        ax1.set_title('Horizontal Motion', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Translation Y
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(frame_nums, translations_y, 'o-', linewidth=2, markersize=6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Frame Index', fontweight='bold')
        ax2.set_ylabel('Translation Y (pixels)', fontweight='bold')
        ax2.set_title('Vertical Motion', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Rotation
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(frame_nums, rotations, 'o-', linewidth=2, markersize=6, color='orange')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Frame Index', fontweight='bold')
        ax3.set_ylabel('Rotation (degrees)', fontweight='bold')
        ax3.set_title('Rotational Motion', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Motion magnitude
        ax4 = fig.add_subplot(gs[1, 0])
        motion_magnitude = np.sqrt(np.array(translations_x)**2 + np.array(translations_y)**2)
        ax4.plot(frame_nums, motion_magnitude, 'o-', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Frame Index', fontweight='bold')
        ax4.set_ylabel('Motion Magnitude (pixels)', fontweight='bold')
        ax4.set_title('Total Motion Displacement', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Scale changes
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(frame_nums, scales, 'o-', linewidth=2, markersize=6, color='brown')
        ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Frame Index', fontweight='bold')
        ax5.set_ylabel('Scale Factor', fontweight='bold')
        ax5.set_title('Scale Variations', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Motion trajectory
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(translations_x, translations_y, 'o-', linewidth=2, markersize=6, color='red')
        ax6.plot(0, 0, 'g*', markersize=20, label='Reference (Mask)')
        ax6.set_xlabel('Translation X (pixels)', fontweight='bold')
        ax6.set_ylabel('Translation Y (pixels)', fontweight='bold')
        ax6.set_title('2D Motion Trajectory', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.axis('equal')
        
        plt.suptitle('Motion Pattern Analysis', fontsize=16, fontweight='bold')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_frame_metrics(self, output_file: Path):
        """Plot comprehensive per-frame metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = self.results['quality_metrics']['per_frame']
        frame_indices = [m['frame_idx'] for m in metrics]
        
        # Extract all metrics
        mi_values = [m['mutual_information'] for m in metrics]
        ssim_values = [m['ssim'] for m in metrics]
        sharpness_values = [m['edge_sharpness'] for m in metrics]
        
        # Heatmap-style visualization
        ax1 = axes[0, 0]
        metrics_matrix = np.array([mi_values, ssim_values, sharpness_values])
        
        # Normalize each row for visualization
        metrics_normalized = np.zeros_like(metrics_matrix)
        for i in range(metrics_matrix.shape[0]):
            row_min = metrics_matrix[i].min()
            row_max = metrics_matrix[i].max()
            if row_max > row_min:
                metrics_normalized[i] = (metrics_matrix[i] - row_min) / (row_max - row_min)
        
        im1 = ax1.imshow(metrics_normalized, aspect='auto', cmap='RdYlGn')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['MI', 'SSIM', 'Sharpness'])
        ax1.set_xlabel('Frame Index', fontweight='bold')
        ax1.set_title('Normalized Metrics Heatmap', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Normalized Value')
        
        # Box plots
        ax2 = axes[0, 1]
        bp = ax2.boxplot([mi_values, ssim_values, 
                          [s/100 for s in sharpness_values]],  # Scale sharpness
                         labels=['MI', 'SSIM', 'Sharp/100'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
            patch.set_facecolor(color)
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Metrics Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Correlation scatter: SSIM vs MI
        ax3 = axes[1, 0]
        ax3.scatter(mi_values, ssim_values, s=100, alpha=0.6, c=frame_indices, cmap='viridis')
        ax3.set_xlabel('Mutual Information', fontweight='bold')
        ax3.set_ylabel('SSIM', fontweight='bold')
        ax3.set_title('SSIM vs MI Correlation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Calculate correlation
        corr = np.corrcoef(mi_values, ssim_values)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax3.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Temporal consistency
        ax4 = axes[1, 1]
        # Calculate frame-to-frame differences
        ssim_diff = np.abs(np.diff(ssim_values))
        mi_diff = np.abs(np.diff(mi_values))
        
        ax4.plot(frame_indices[1:], ssim_diff, 'o-', label='SSIM Δ', linewidth=2)
        ax4.plot(frame_indices[1:], mi_diff/max(mi_diff)*max(ssim_diff), 's-', 
                label='MI Δ (scaled)', linewidth=2)
        ax4.set_xlabel('Frame Index', fontweight='bold')
        ax4.set_ylabel('Frame-to-Frame Change', fontweight='bold')
        ax4.set_title('Temporal Consistency', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Frame Metrics Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_difference_maps(self, output_file: Path):
        """Create difference maps showing registration improvement"""
        fig = plt.figure(figsize=(20, 10))
        
        mask_idx = self.results['mask_frame_idx']
        mask_original = self.results['preprocessed_frames'][mask_idx]
        mask_registered = self.results['registered_frames'][mask_idx]
        
        # Select 6 frames to compare
        frame_indices = [i for i in range(len(self.results['raw_frames'])) if i != mask_idx]
        display_indices = frame_indices[:min(6, len(frame_indices))]
        
        for idx, frame_num in enumerate(display_indices):
            # Before registration difference
            ax1 = plt.subplot(len(display_indices), 3, idx*3 + 1)
            diff_before = cv2.absdiff(
                self.results['preprocessed_frames'][frame_num],
                mask_original
            )
            im1 = ax1.imshow(diff_before, cmap='jet', vmin=0, vmax=50)
            ax1.set_title(f'Frame {frame_num} - Before', fontweight='bold', fontsize=10)
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # After registration difference
            ax2 = plt.subplot(len(display_indices), 3, idx*3 + 2)
            diff_after = cv2.absdiff(
                self.results['registered_frames'][frame_num],
                mask_registered
            )
            im2 = ax2.imshow(diff_after, cmap='jet', vmin=0, vmax=50)
            ax2.set_title(f'Frame {frame_num} - After', fontweight='bold', fontsize=10)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # Improvement map
            ax3 = plt.subplot(len(display_indices), 3, idx*3 + 3)
            improvement = diff_before.astype(np.float32) - diff_after.astype(np.float32)
            im3 = ax3.imshow(improvement, cmap='RdYlGn', vmin=-10, vmax=30)
            ax3.set_title(f'Improvement (Green=Better)', fontweight='bold', fontsize=10)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        plt.suptitle('Registration Improvement - Difference Maps', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_animations(self, output_path: Path):
        """Create animated GIFs for visual comparison"""
        
        # Animation 1: Original sequence
        self._create_animation(
            self.results['preprocessed_frames'],
            output_path / "animation_original.gif",
            "Original Sequence"
        )
        
        # Animation 2: Registered sequence
        self._create_animation(
            self.results['registered_frames'],
            output_path / "animation_registered.gif",
            "Registered Sequence"
        )
        
        # Animation 3: DSA sequence
        self._create_animation(
            self.results['dsa_images'],
            output_path / "animation_dsa.gif",
            "DSA Sequence",
            cmap='hot'
        )
    
    def _create_animation(self, frames: List[np.ndarray], output_file: Path,
                         title: str, cmap: str = 'gray'):
        """Create animated GIF from frame sequence"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Initialize with first frame
        im = ax.imshow(frames[0], cmap=cmap, animated=True)
        ax.axis('off')
        title_text = ax.set_title(f'{title} - Frame 0', fontweight='bold', fontsize=14)
        
        def update(frame_num):
            im.set_array(frames[frame_num])
            title_text.set_text(f'{title} - Frame {frame_num}')
            return [im, title_text]
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(frames),
            interval=200, blit=True, repeat=True
        )
        
        # Save as GIF
        ani.save(output_file, writer='pillow', fps=5, dpi=100)
        plt.close()
        print(f"Animation saved: {output_file}")
    
    def create_side_by_side_video(self, output_path: Path):
        """Create side-by-side comparison video"""
        frames_original = self.results['preprocessed_frames']
        frames_registered = self.results['registered_frames']
        frames_dsa = self.results['dsa_images']
        
        # Create figure for side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(frames_original[0], cmap='gray')
        axes[0].set_title('Original', fontweight='bold', fontsize=14)
        axes[0].axis('off')
        
        im2 = axes[1].imshow(frames_registered[0], cmap='gray')
        axes[1].set_title('Registered', fontweight='bold', fontsize=14)
        axes[1].axis('off')
        
        im3 = axes[2].imshow(frames_dsa[0], cmap='hot')
        axes[2].set_title('DSA', fontweight='bold', fontsize=14)
        axes[2].axis('off')
        
        frame_text = fig.suptitle('Frame 0', fontsize=16, fontweight='bold')
        
        def update(frame_num):
            im1.set_array(frames_original[frame_num])
            im2.set_array(frames_registered[frame_num])
            im3.set_array(frames_dsa[frame_num])
            frame_text.set_text(f'Frame {frame_num}')
            return [im1, im2, im3, frame_text]
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(frames_original),
            interval=200, blit=True, repeat=True
        )
        
        output_file = output_path / "comparison_video.gif"
        ani.save(output_file, writer='pillow', fps=5, dpi=100)
        plt.close()
        print(f"Comparison video saved: {output_file}")


def visualize_results(results: Dict, output_dir: str):
    """
    Main function to generate all visualizations
    
    Args:
        results: Results dictionary from DSAProcessor
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = DSAVisualizer(results)
    visualizer.create_comprehensive_report(output_path)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"All visualizations saved to: {output_path}")
    print("="*70)


# Example usage with the main pipeline
if __name__ == "__main__":
    """
    Complete workflow example:
    1. Process DICOM sequence
    2. Generate visualizations
    """
    
    # Import from main pipeline
    from dsa_motion_correction import main_pipeline
    
    # Process sequence
    # results = main_pipeline(
    #     input_path="/path/to/dicom/sequence",
    #     output_dir="/path/to/output",
    #     anatomy_type="peripheral"  # or 'neuro', 'gi'
    # )
    
    # Generate visualizations
    # visualize_results(results, "/path/to/output")
    
    print("DSA Visualization Tools Ready")