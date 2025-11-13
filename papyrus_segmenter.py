"""
Advanced surface segmentation for Vesuvius Challenge.

This module handles papyrus-specific surface detection:
- Recto (horizontal fibers) vs Verso (vertical fibers) detection
- Topology-aware processing to avoid mergers and artificial holes
- Post-processing to ensure proper surface connectivity
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, filters
import matplotlib.pyplot as plt


class PapyrusSegmenter:
    """
    Papyrus-specific surface segmenter.
    
    Handles the unique challenges of papyrus scrolls:
    - Two-layer structure (recto/verso)
    - Damaged and frayed sheets
    - Topological correctness (no mergers, no artificial holes)
    """
    
    def __init__(self, 
                 min_component_size=100,
                 max_hole_size=50,
                 merge_distance_threshold=5):
        """
        Initialize papyrus segmenter.
        
        Args:
            min_component_size (int): Minimum size for valid surface components
            max_hole_size (int): Maximum size of holes to fill
            merge_distance_threshold (int): Minimum distance between separate sheets
        """
        self.min_component_size = min_component_size
        self.max_hole_size = max_hole_size
        self.merge_distance_threshold = merge_distance_threshold
    
    def detect_fiber_orientation(self, ct_slice, window_size=15):
        """
        Detect fiber orientation to distinguish recto (horizontal) from verso (vertical).
        
        Args:
            ct_slice (numpy.ndarray): CT scan slice
            window_size (int): Size of analysis window
            
        Returns:
            numpy.ndarray: Orientation map (0=horizontal/recto, 1=vertical/verso)
        """
        # Compute gradients
        grad_y = filters.sobel_v(ct_slice)
        grad_x = filters.sobel_h(ct_slice)
        
        # Orientation angle
        orientation = np.arctan2(grad_y, grad_x)
        
        # Classify: horizontal fibers (recto) have vertical edges
        # vertical fibers (verso) have horizontal edges
        horizontal_strength = np.abs(np.sin(orientation))  # Vertical edges
        vertical_strength = np.abs(np.cos(orientation))    # Horizontal edges
        
        # Apply smoothing
        horizontal_strength = ndimage.gaussian_filter(horizontal_strength, sigma=2)
        vertical_strength = ndimage.gaussian_filter(vertical_strength, sigma=2)
        
        # Classify: 0 = recto (horizontal fibers), 1 = verso (vertical fibers)
        orientation_map = (vertical_strength > horizontal_strength).astype(np.uint8)
        
        return orientation_map
    
    def segment_with_topology(self, probability_map, threshold=0.5):
        """
        Segment surface with topological constraints.
        
        Args:
            probability_map (numpy.ndarray): Surface probability map
            threshold (float): Threshold for binary segmentation
            
        Returns:
            numpy.ndarray: Binary segmentation with topology preserved
        """
        # Initial thresholding
        binary = (probability_map > threshold).astype(np.uint8)
        
        # Step 1: Remove small components (noise)
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        
        for prop in props:
            if prop.area < self.min_component_size:
                binary[labeled == prop.label] = 0
        
        # Step 2: Fill small holes (avoid artificial disconnections)
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
        
        # Remove holes that are too large (real gaps between sheets)
        labeled_holes = measure.label(1 - binary)
        hole_props = measure.regionprops(labeled_holes)
        
        for prop in hole_props:
            if prop.area > self.max_hole_size:
                # Keep this as a hole (don't fill)
                coords = prop.coords
                binary[coords[:, 0], coords[:, 1]] = 0
        
        # Step 3: Prevent mergers - check distance between components
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        
        if len(props) > 1:
            # Check for components that are too close (potential mergers)
            for i, prop1 in enumerate(props):
                for prop2 in props[i+1:]:
                    # Compute minimum distance between components
                    min_dist = self._min_distance_between_regions(
                        prop1.coords, prop2.coords
                    )
                    
                    if min_dist < self.merge_distance_threshold:
                        # Components are too close - may be incorrectly merged
                        # Apply morphological opening to separate
                        binary = morphology.binary_opening(
                            binary, 
                            morphology.disk(self.merge_distance_threshold // 2)
                        ).astype(np.uint8)
                        break
        
        return binary
    
    def _min_distance_between_regions(self, coords1, coords2, max_samples=1000):
        """
        Compute minimum distance between two regions.
        
        Args:
            coords1, coords2: Coordinate arrays for regions
            max_samples: Maximum number of points to sample for efficiency
            
        Returns:
            float: Minimum distance
        """
        # Sample points if regions are large
        if len(coords1) > max_samples:
            indices = np.random.choice(len(coords1), max_samples, replace=False)
            coords1 = coords1[indices]
        
        if len(coords2) > max_samples:
            indices = np.random.choice(len(coords2), max_samples, replace=False)
            coords2 = coords2[indices]
        
        # Compute pairwise distances
        min_dist = float('inf')
        for c1 in coords1[::10]:  # Sample every 10th point
            dists = np.sqrt(np.sum((coords2 - c1)**2, axis=1))
            min_dist = min(min_dist, dists.min())
        
        return min_dist
    
    def post_process_segmentation(self, segmentation, connectivity_check=True):
        """
        Post-process segmentation to ensure quality.
        
        Args:
            segmentation (numpy.ndarray): Binary segmentation
            connectivity_check (bool): Whether to check and report connectivity issues
            
        Returns:
            dict: Processed segmentation and statistics
        """
        # Label connected components
        labeled = measure.label(segmentation)
        num_components = labeled.max()
        
        props = measure.regionprops(labeled)
        
        # Compute statistics
        stats = {
            'num_components': num_components,
            'component_sizes': [prop.area for prop in props],
            'total_surface_pixels': segmentation.sum(),
            'coverage': segmentation.sum() / segmentation.size
        }
        
        # Check for topology issues
        if connectivity_check:
            issues = []
            
            # Check for very small components (noise)
            small_components = [p.area for p in props if p.area < self.min_component_size]
            if small_components:
                issues.append(f"Found {len(small_components)} small components (possible noise)")
            
            # Check for very large holes
            holes = ndimage.binary_fill_holes(segmentation) - segmentation
            if holes.sum() > 0:
                hole_labeled = measure.label(holes)
                hole_props = measure.regionprops(hole_labeled)
                large_holes = [p.area for p in hole_props if p.area > self.max_hole_size]
                if large_holes:
                    issues.append(f"Found {len(large_holes)} large holes (possible disconnections)")
            
            stats['topology_issues'] = issues
        
        # Apply final morphological cleanup
        cleaned = morphology.remove_small_objects(
            segmentation.astype(bool), 
            min_size=self.min_component_size
        ).astype(np.uint8)
        
        return {
            'segmentation': cleaned,
            'labeled': measure.label(cleaned),
            'stats': stats
        }
    
    def segment_papyrus_sheet(self, ct_slice, probability_map, 
                            prefer_recto=True, threshold=0.5):
        """
        Complete papyrus sheet segmentation pipeline.
        
        Args:
            ct_slice (numpy.ndarray): CT scan slice
            probability_map (numpy.ndarray): Surface probability map from model
            prefer_recto (bool): Prefer recto surface when both layers visible
            threshold (float): Probability threshold
            
        Returns:
            dict: Complete segmentation results
        """
        # Detect fiber orientation
        orientation_map = self.detect_fiber_orientation(ct_slice)
        
        # Segment with topology preservation
        segmentation = self.segment_with_topology(probability_map, threshold)
        
        # If prefer_recto, weight towards horizontal fibers
        if prefer_recto:
            # Boost regions with horizontal fibers (recto)
            recto_mask = (orientation_map == 0)
            boosted_prob = probability_map.copy()
            boosted_prob[recto_mask] *= 1.2  # Boost recto probabilities
            boosted_prob = np.clip(boosted_prob, 0, 1)
            
            segmentation = self.segment_with_topology(boosted_prob, threshold)
        
        # Post-process
        result = self.post_process_segmentation(segmentation)
        
        # Add orientation info
        result['orientation_map'] = orientation_map
        result['recto_pixels'] = np.sum(segmentation & (orientation_map == 0))
        result['verso_pixels'] = np.sum(segmentation & (orientation_map == 1))
        
        return result
    
    def visualize_segmentation(self, ct_slice, result, title="Papyrus Segmentation"):
        """
        Visualize segmentation results with topology information.
        
        Args:
            ct_slice (numpy.ndarray): Original CT slice
            result (dict): Segmentation result from segment_papyrus_sheet
            title (str): Plot title
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original CT slice
        axes[0, 0].imshow(ct_slice, cmap='gray')
        axes[0, 0].set_title('Original CT Slice')
        axes[0, 0].axis('off')
        
        # Fiber orientation
        axes[0, 1].imshow(result['orientation_map'], cmap='RdYlBu')
        axes[0, 1].set_title('Fiber Orientation\n(Blue=Recto/Horizontal, Red=Verso/Vertical)')
        axes[0, 1].axis('off')
        
        # Binary segmentation
        axes[0, 2].imshow(result['segmentation'], cmap='hot')
        axes[0, 2].set_title(f"Segmentation\n({result['stats']['num_components']} components)")
        axes[0, 2].axis('off')
        
        # Labeled components
        axes[1, 0].imshow(result['labeled'], cmap='tab20')
        axes[1, 0].set_title('Connected Components')
        axes[1, 0].axis('off')
        
        # Overlay on CT
        axes[1, 1].imshow(ct_slice, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(result['segmentation'], cmap='hot', alpha=0.3)
        axes[1, 1].set_title('Overlay')
        axes[1, 1].axis('off')
        
        # Statistics text
        axes[1, 2].axis('off')
        stats_text = f"Segmentation Statistics:\n\n"
        stats_text += f"Components: {result['stats']['num_components']}\n"
        stats_text += f"Total pixels: {result['stats']['total_surface_pixels']}\n"
        stats_text += f"Coverage: {result['stats']['coverage']:.2%}\n\n"
        stats_text += f"Recto pixels: {result['recto_pixels']}\n"
        stats_text += f"Verso pixels: {result['verso_pixels']}\n\n"
        
        if 'topology_issues' in result['stats'] and result['stats']['topology_issues']:
            stats_text += "⚠️ Topology Issues:\n"
            for issue in result['stats']['topology_issues']:
                stats_text += f"  • {issue}\n"
        else:
            stats_text += "✓ No topology issues"
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def evaluate_segmentation_quality(predicted, ground_truth):
    """
    Evaluate segmentation quality against ground truth.
    
    Args:
        predicted (numpy.ndarray): Predicted binary segmentation
        ground_truth (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Ensure same shape
    if predicted.shape != ground_truth.shape:
        raise ValueError("Predicted and ground truth must have same shape")
    
    # Flatten arrays
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Compute metrics
    true_positive = np.sum((pred_flat == 1) & (gt_flat == 1))
    false_positive = np.sum((pred_flat == 1) & (gt_flat == 0))
    true_negative = np.sum((pred_flat == 0) & (gt_flat == 0))
    false_negative = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    # Standard metrics
    accuracy = (true_positive + true_negative) / len(pred_flat)
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU (Intersection over Union) - Jaccard index
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice': dice,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative
    }


if __name__ == "__main__":
    print("=== Papyrus Segmenter Demo ===\n")
    
    from load_ct_scans import create_sample_data, load_ct_slice
    
    # Create sample data
    print("Creating sample data...")
    sample_folder = "/tmp/sample_ct_scans"
    create_sample_data(sample_folder, num_slices=5)
    
    # Load a slice
    ct_slice = load_ct_slice(f"{sample_folder}/slice_0002.png")
    
    # Create a mock probability map (using edge detection as proxy)
    from surface_tracker import SimpleSurfaceTracker
    tracker = SimpleSurfaceTracker()
    result = tracker.track_surface_in_slice(ct_slice)
    
    # Normalize edges to [0, 1] as probability
    prob_map = result['edges'] / result['edges'].max()
    
    # Initialize segmenter
    segmenter = PapyrusSegmenter(
        min_component_size=100,
        max_hole_size=50,
        merge_distance_threshold=5
    )
    
    # Segment
    print("\nSegmenting papyrus sheet...")
    seg_result = segmenter.segment_papyrus_sheet(
        ct_slice, 
        prob_map, 
        prefer_recto=True, 
        threshold=0.3
    )
    
    print(f"\nResults:")
    print(f"  Components: {seg_result['stats']['num_components']}")
    print(f"  Total pixels: {seg_result['stats']['total_surface_pixels']}")
    print(f"  Recto pixels: {seg_result['recto_pixels']}")
    print(f"  Verso pixels: {seg_result['verso_pixels']}")
    
    if seg_result['stats'].get('topology_issues'):
        print(f"\n⚠️  Topology issues detected:")
        for issue in seg_result['stats']['topology_issues']:
            print(f"    - {issue}")
    else:
        print(f"\n✓ No topology issues detected")
    
    print("\n✓ Demo complete! Use visualize_segmentation() to see results.")
