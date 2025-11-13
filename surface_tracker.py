"""
Surface tracking for CT scan volumes.

This module implements a simple surface tracking algorithm to follow
the surface of a scroll in CT scan data. This is essential for virtually
unwrapping ancient scrolls like those from Villa dei Papiri.
"""

import numpy as np
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt


class SimpleSurfaceTracker:
    """
    A simple surface tracker that identifies and follows surfaces in CT volumes.
    
    This uses basic image processing techniques suitable for beginners:
    - Edge detection to find surfaces
    - Gradient following to track surface continuity
    """
    
    def __init__(self, smoothing_sigma=1.0, threshold_percentile=70):
        """
        Initialize the surface tracker.
        
        Args:
            smoothing_sigma (float): Gaussian smoothing parameter
            threshold_percentile (float): Percentile for thresholding (0-100)
        """
        self.smoothing_sigma = smoothing_sigma
        self.threshold_percentile = threshold_percentile
    
    def detect_edges(self, ct_slice):
        """
        Detect edges in a CT scan slice using Sobel filter.
        
        Args:
            ct_slice (numpy.ndarray): 2D CT scan slice
            
        Returns:
            numpy.ndarray: Edge magnitude map
        """
        # Smooth the image first
        smoothed = ndimage.gaussian_filter(ct_slice.astype(float), 
                                          sigma=self.smoothing_sigma)
        
        # Apply Sobel edge detection
        edges = filters.sobel(smoothed)
        
        return edges
    
    def find_surface_points(self, edges, num_points=100):
        """
        Find strong surface points from edge detection.
        
        Args:
            edges (numpy.ndarray): Edge magnitude map
            num_points (int): Number of surface points to extract
            
        Returns:
            numpy.ndarray: Array of (y, x) coordinates of surface points
        """
        # Threshold to get strong edges
        threshold = np.percentile(edges, self.threshold_percentile)
        strong_edges = edges > threshold
        
        # Get coordinates of strong edge points
        y_coords, x_coords = np.where(strong_edges)
        
        if len(y_coords) == 0:
            return np.array([])
        
        # Sample points uniformly
        if len(y_coords) > num_points:
            indices = np.linspace(0, len(y_coords) - 1, num_points, dtype=int)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        return np.column_stack([y_coords, x_coords])
    
    def track_surface_in_slice(self, ct_slice, num_points=100):
        """
        Track the surface in a single CT scan slice.
        
        Args:
            ct_slice (numpy.ndarray): 2D CT scan slice
            num_points (int): Number of surface points to track
            
        Returns:
            dict: Dictionary containing:
                - 'edges': Edge magnitude map
                - 'surface_points': Array of (y, x) surface coordinates
        """
        # Detect edges
        edges = self.detect_edges(ct_slice)
        
        # Find surface points
        surface_points = self.find_surface_points(edges, num_points)
        
        return {
            'edges': edges,
            'surface_points': surface_points
        }
    
    def track_surface_in_volume(self, volume, num_points=100):
        """
        Track surfaces across all slices in a 3D volume.
        
        Args:
            volume (numpy.ndarray): 3D CT scan volume
            num_points (int): Number of surface points per slice
            
        Returns:
            list: List of tracking results for each slice
        """
        results = []
        
        print(f"Tracking surface across {volume.shape[0]} slices...")
        for i in range(volume.shape[0]):
            result = self.track_surface_in_slice(volume[i], num_points)
            results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{volume.shape[0]} slices")
        
        print("✓ Surface tracking complete")
        return results
    
    def visualize_tracking(self, ct_slice, tracking_result, title="Surface Tracking"):
        """
        Visualize surface tracking results on a CT slice.
        
        Args:
            ct_slice (numpy.ndarray): Original CT scan slice
            tracking_result (dict): Result from track_surface_in_slice
            title (str): Plot title
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original slice
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title('Original CT Slice')
        axes[0].axis('off')
        
        # Edge detection
        axes[1].imshow(tracking_result['edges'], cmap='hot')
        axes[1].set_title('Edge Detection')
        axes[1].axis('off')
        
        # Surface points overlay
        axes[2].imshow(ct_slice, cmap='gray')
        if len(tracking_result['surface_points']) > 0:
            points = tracking_result['surface_points']
            axes[2].scatter(points[:, 1], points[:, 0], c='cyan', s=10, alpha=0.6)
        axes[2].set_title('Detected Surface Points')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class SurfaceFeatureExtractor:
    """
    Extract features from surface points for training ML models.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize feature extractor.
        
        Args:
            window_size (int): Size of local window for feature extraction
        """
        self.window_size = window_size
    
    def extract_local_features(self, ct_slice, point):
        """
        Extract local features around a surface point.
        
        Args:
            ct_slice (numpy.ndarray): CT scan slice
            point (tuple): (y, x) coordinate
            
        Returns:
            numpy.ndarray: Feature vector
        """
        y, x = int(point[0]), int(point[1])
        half_w = self.window_size // 2
        
        # Get local patch (with boundary handling)
        y_min = max(0, y - half_w)
        y_max = min(ct_slice.shape[0], y + half_w + 1)
        x_min = max(0, x - half_w)
        x_max = min(ct_slice.shape[1], x + half_w + 1)
        
        patch = ct_slice[y_min:y_max, x_min:x_max]
        
        # Compute features
        features = [
            np.mean(patch),           # Mean intensity
            np.std(patch),            # Standard deviation
            np.median(patch),         # Median intensity
            np.min(patch),            # Min intensity
            np.max(patch),            # Max intensity
            patch.max() - patch.min() # Intensity range
        ]
        
        return np.array(features)
    
    def extract_features_from_points(self, ct_slice, surface_points):
        """
        Extract features for all surface points.
        
        Args:
            ct_slice (numpy.ndarray): CT scan slice
            surface_points (numpy.ndarray): Array of (y, x) coordinates
            
        Returns:
            numpy.ndarray: Feature matrix (num_points, num_features)
        """
        if len(surface_points) == 0:
            return np.array([])
        
        features = []
        for point in surface_points:
            feat = self.extract_local_features(ct_slice, point)
            features.append(feat)
        
        return np.array(features)


if __name__ == "__main__":
    # Demo: Load sample data and track surface
    print("=== Surface Tracker Demo ===\n")
    
    # Create sample data first
    from load_ct_scans import create_sample_data, load_ct_volume, load_ct_slice
    
    sample_folder = "/tmp/sample_ct_scans"
    create_sample_data(sample_folder, num_slices=10)
    
    # Load a single slice
    print("\n--- Tracking surface in single slice ---")
    ct_slice = load_ct_slice(f"{sample_folder}/slice_0005.png")
    
    # Initialize tracker
    tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)
    
    # Track surface
    result = tracker.track_surface_in_slice(ct_slice, num_points=150)
    print(f"Found {len(result['surface_points'])} surface points")
    
    # Extract features
    print("\n--- Extracting features ---")
    feature_extractor = SurfaceFeatureExtractor(window_size=5)
    features = feature_extractor.extract_features_from_points(ct_slice, result['surface_points'])
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature statistics:\n  Mean: {features.mean(axis=0)}\n  Std: {features.std(axis=0)}")
    
    print("\n✓ Demo complete! Use visualize_tracking() to see results.")
