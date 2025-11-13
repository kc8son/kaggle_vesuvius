"""
Load and visualize CT scan data from the Vesuvius Challenge.

This module provides simple utilities for loading CT scan slices and 
preparing them for surface tracking training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os


def load_ct_slice(file_path):
    """
    Load a single CT scan slice from an image file.
    
    Args:
        file_path (str or Path): Path to the CT scan image file
        
    Returns:
        numpy.ndarray: 2D array representing the CT scan slice
    """
    img = Image.open(file_path)
    return np.array(img)


def load_ct_volume(folder_path, start_slice=0, end_slice=None, step=1):
    """
    Load multiple CT scan slices to form a 3D volume.
    
    Args:
        folder_path (str or Path): Path to folder containing CT scan slices
        start_slice (int): Index of first slice to load
        end_slice (int): Index of last slice to load (None for all)
        step (int): Step size between slices
        
    Returns:
        numpy.ndarray: 3D array of shape (num_slices, height, width)
    """
    folder = Path(folder_path)
    
    # Find all image files
    image_files = sorted([
        f for f in folder.glob('*.tif*') 
    ] + [
        f for f in folder.glob('*.png')
    ] + [
        f for f in folder.glob('*.jpg')
    ])
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    # Select slice range
    if end_slice is None:
        end_slice = len(image_files)
    
    selected_files = image_files[start_slice:end_slice:step]
    
    # Load first image to get dimensions
    first_img = load_ct_slice(selected_files[0])
    height, width = first_img.shape[:2]
    
    # Pre-allocate volume array
    volume = np.zeros((len(selected_files), height, width), dtype=first_img.dtype)
    
    # Load all slices
    print(f"Loading {len(selected_files)} CT scan slices...")
    for i, file_path in enumerate(selected_files):
        img = load_ct_slice(file_path)
        if img.ndim == 3:  # Convert RGB to grayscale if needed
            img = img.mean(axis=2).astype(img.dtype)
        volume[i] = img
        
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(selected_files)} slices")
    
    print(f"✓ Volume loaded: shape {volume.shape}")
    return volume


def visualize_slice(ct_slice, title="CT Scan Slice", cmap='gray'):
    """
    Visualize a single CT scan slice.
    
    Args:
        ct_slice (numpy.ndarray): 2D array of CT scan data
        title (str): Title for the plot
        cmap (str): Colormap to use
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(ct_slice, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_volume_slice(volume, slice_idx, axis=0, title=None, cmap='gray'):
    """
    Visualize a slice from a 3D CT volume along a specified axis.
    
    Args:
        volume (numpy.ndarray): 3D array of CT scan data
        slice_idx (int): Index of slice to display
        axis (int): Axis along which to slice (0=depth, 1=height, 2=width)
        title (str): Title for the plot
        cmap (str): Colormap to use
    """
    if axis == 0:
        slice_data = volume[slice_idx, :, :]
        default_title = f"Slice {slice_idx} (depth axis)"
    elif axis == 1:
        slice_data = volume[:, slice_idx, :]
        default_title = f"Slice {slice_idx} (height axis)"
    else:
        slice_data = volume[:, :, slice_idx]
        default_title = f"Slice {slice_idx} (width axis)"
    
    visualize_slice(slice_data, title=title or default_title, cmap=cmap)


def create_sample_data(output_folder, num_slices=20, size=(512, 512)):
    """
    Create sample CT scan data for testing (simulated scroll-like structure).
    
    Args:
        output_folder (str or Path): Folder to save sample images
        num_slices (int): Number of slices to generate
        size (tuple): Size of each slice (height, width)
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_slices} sample CT scan slices...")
    
    for i in range(num_slices):
        # Create a simulated scroll cross-section with spiral pattern
        y, x = np.ogrid[:size[0], :size[1]]
        center_y, center_x = size[0] // 2, size[1] // 2
        
        # Distance from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        theta = np.arctan2(y - center_y, x - center_x)
        
        # Create spiral pattern (simulating scroll layers)
        spiral = np.sin(r / 20 + theta * 3 + i / 5)
        
        # Add noise
        noise = np.random.randn(size[0], size[1]) * 0.1
        
        # Combine and normalize
        img_data = spiral + noise
        img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255)
        img_data = img_data.astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_data)
        img.save(output_path / f"slice_{i:04d}.png")
    
    print(f"✓ Sample data created in {output_folder}")


if __name__ == "__main__":
    # Demo: Create and load sample data
    print("=== CT Scan Loader Demo ===\n")
    
    # Create sample data
    sample_folder = "/tmp/sample_ct_scans"
    create_sample_data(sample_folder, num_slices=20)
    
    # Load single slice
    print("\n--- Loading single slice ---")
    single_slice = load_ct_slice(f"{sample_folder}/slice_0010.png")
    print(f"Slice shape: {single_slice.shape}")
    print(f"Slice dtype: {single_slice.dtype}")
    print(f"Value range: [{single_slice.min()}, {single_slice.max()}]")
    
    # Load volume
    print("\n--- Loading volume ---")
    volume = load_ct_volume(sample_folder, start_slice=0, end_slice=15, step=1)
    print(f"Volume shape: {volume.shape}")
    
    print("\n✓ Demo complete! Use visualize_slice() or visualize_volume_slice() to view the data.")
