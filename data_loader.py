"""
Data loader for Vesuvius Challenge dataset.

This module handles the official dataset format with 3D chunks of binary labeled
CT scans from Herculaneum scrolls. Supports variable chunk dimensions and
ground truth labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


class VesuviusDataLoader:
    """
    Loader for Vesuvius Challenge dataset with binary labels.
    
    Handles:
    - Variable chunk dimensions
    - Binary surface labels (ground truth)
    - Data from ESRF (Grenoble) and DLS (Oxford) synchrotrons
    """
    
    def __init__(self, data_root):
        """
        Initialize the data loader.
        
        Args:
            data_root (str or Path): Root directory of Vesuvius dataset
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {data_root}")
    
    def load_chunk(self, chunk_id, load_labels=True):
        """
        Load a single 3D chunk with optional labels.
        
        Args:
            chunk_id (str): Identifier for the chunk (e.g., "scroll1_chunk1")
            load_labels (bool): Whether to load binary labels
            
        Returns:
            dict: Dictionary containing:
                - 'volume': 3D numpy array of CT data
                - 'labels': 3D binary array (if load_labels=True)
                - 'metadata': Dictionary with chunk information
        """
        chunk_path = self.data_root / chunk_id
        
        if not chunk_path.exists():
            raise ValueError(f"Chunk not found: {chunk_path}")
        
        # Load volume data
        volume_path = chunk_path / "surface_volume"
        if not volume_path.exists():
            volume_path = chunk_path / "volume"
        
        if not volume_path.exists():
            raise ValueError(f"Volume directory not found in {chunk_path}")
        
        volume = self._load_volume_from_dir(volume_path)
        
        result = {
            'volume': volume,
            'metadata': {
                'chunk_id': chunk_id,
                'shape': volume.shape,
                'dtype': str(volume.dtype)
            }
        }
        
        # Load labels if requested
        if load_labels:
            labels_path = chunk_path / "inklabels.png"
            if not labels_path.exists():
                labels_path = chunk_path / "labels.png"
            if not labels_path.exists():
                labels_path = chunk_path / "mask.png"
            
            if labels_path.exists():
                labels = np.array(Image.open(labels_path))
                # Convert to binary (0 or 1)
                labels = (labels > 0).astype(np.uint8)
                result['labels'] = labels
                result['metadata']['has_labels'] = True
            else:
                print(f"Warning: No labels found for {chunk_id}")
                result['labels'] = None
                result['metadata']['has_labels'] = False
        
        # Load metadata if available
        metadata_path = chunk_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                extra_metadata = json.load(f)
                result['metadata'].update(extra_metadata)
        
        return result
    
    def _load_volume_from_dir(self, volume_dir):
        """
        Load 3D volume from directory of slice images.
        
        Args:
            volume_dir (Path): Directory containing slice images
            
        Returns:
            numpy.ndarray: 3D volume array
        """
        # Find all image files
        image_files = sorted(list(volume_dir.glob('*.tif*')) + 
                           list(volume_dir.glob('*.png')) +
                           list(volume_dir.glob('*.jpg')))
        
        if not image_files:
            raise ValueError(f"No image files found in {volume_dir}")
        
        # Load first slice to get dimensions
        first_slice = np.array(Image.open(image_files[0]))
        if first_slice.ndim == 3:
            first_slice = first_slice.mean(axis=2).astype(first_slice.dtype)
        
        # Pre-allocate volume
        volume = np.zeros((len(image_files), *first_slice.shape), dtype=first_slice.dtype)
        
        # Load all slices
        for i, img_path in enumerate(tqdm(image_files, desc="Loading slices", leave=False)):
            img = np.array(Image.open(img_path))
            if img.ndim == 3:
                img = img.mean(axis=2).astype(img.dtype)
            volume[i] = img
        
        return volume
    
    def list_chunks(self):
        """
        List all available chunks in the dataset.
        
        Returns:
            list: List of chunk identifiers
        """
        chunks = []
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Check if it contains volume data
                if (item / "surface_volume").exists() or (item / "volume").exists():
                    chunks.append(item.name)
        
        return sorted(chunks)
    
    def get_chunk_info(self, chunk_id):
        """
        Get information about a chunk without loading the full data.
        
        Args:
            chunk_id (str): Chunk identifier
            
        Returns:
            dict: Chunk information including dimensions and label availability
        """
        chunk_path = self.data_root / chunk_id
        
        # Find volume directory
        volume_path = chunk_path / "surface_volume"
        if not volume_path.exists():
            volume_path = chunk_path / "volume"
        
        if not volume_path.exists():
            raise ValueError(f"Volume directory not found in {chunk_path}")
        
        # Count slices
        image_files = (list(volume_path.glob('*.tif*')) + 
                      list(volume_path.glob('*.png')) +
                      list(volume_path.glob('*.jpg')))
        num_slices = len(image_files)
        
        # Get dimensions from first slice
        if num_slices > 0:
            first_img = np.array(Image.open(image_files[0]))
            if first_img.ndim == 3:
                first_img = first_img.mean(axis=2)
            height, width = first_img.shape
        else:
            height, width = 0, 0
        
        # Check for labels
        has_labels = False
        for label_name in ['inklabels.png', 'labels.png', 'mask.png']:
            if (chunk_path / label_name).exists():
                has_labels = True
                break
        
        return {
            'chunk_id': chunk_id,
            'shape': (num_slices, height, width),
            'has_labels': has_labels,
            'path': str(chunk_path)
        }


class LabeledDataset:
    """
    Dataset wrapper for training with labeled data.
    """
    
    def __init__(self, loader, chunk_ids):
        """
        Initialize labeled dataset.
        
        Args:
            loader (VesuviusDataLoader): Data loader instance
            chunk_ids (list): List of chunk IDs to include
        """
        self.loader = loader
        self.chunk_ids = chunk_ids
        self.chunks = []
        
        # Load all chunks
        print(f"Loading {len(chunk_ids)} chunks...")
        for chunk_id in chunk_ids:
            chunk_data = loader.load_chunk(chunk_id, load_labels=True)
            if chunk_data['labels'] is not None:
                self.chunks.append(chunk_data)
            else:
                print(f"Skipping {chunk_id} - no labels")
        
        print(f"✓ Loaded {len(self.chunks)} chunks with labels")
    
    def get_training_samples(self, num_samples_per_chunk=1000, balance=True):
        """
        Extract training samples from labeled chunks.
        
        Args:
            num_samples_per_chunk (int): Number of samples per chunk
            balance (bool): Whether to balance positive/negative samples
            
        Returns:
            tuple: (features, labels) for training
        """
        from surface_tracker import SurfaceFeatureExtractor
        
        extractor = SurfaceFeatureExtractor(window_size=5)
        
        all_features = []
        all_labels = []
        
        print(f"Extracting training samples from {len(self.chunks)} chunks...")
        
        for chunk_data in self.chunks:
            volume = chunk_data['volume']
            labels = chunk_data['labels']
            
            # Sample from middle slice (most representative)
            mid_idx = volume.shape[0] // 2
            ct_slice = volume[mid_idx]
            
            # Get positive samples (surface points)
            positive_coords = np.argwhere(labels > 0)
            
            if len(positive_coords) == 0:
                print(f"Warning: No positive samples in {chunk_data['metadata']['chunk_id']}")
                continue
            
            # Sample positive points
            num_positive = min(num_samples_per_chunk // 2, len(positive_coords))
            if len(positive_coords) > num_positive:
                pos_indices = np.random.choice(len(positive_coords), num_positive, replace=False)
                positive_coords = positive_coords[pos_indices]
            
            # Get negative samples (background points)
            negative_coords = np.argwhere(labels == 0)
            num_negative = min(num_samples_per_chunk // 2 if balance else num_samples_per_chunk, 
                             len(negative_coords))
            if len(negative_coords) > num_negative:
                neg_indices = np.random.choice(len(negative_coords), num_negative, replace=False)
                negative_coords = negative_coords[neg_indices]
            
            # Extract features
            for coords in positive_coords:
                features = extractor.extract_local_features(ct_slice, coords)
                all_features.append(features)
                all_labels.append(1)
            
            for coords in negative_coords:
                features = extractor.extract_local_features(ct_slice, coords)
                all_features.append(features)
                all_labels.append(0)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"✓ Extracted {len(X)} samples")
        print(f"  Positive: {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
        print(f"  Negative: {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")
        
        return X, y


def visualize_chunk_with_labels(chunk_data, slice_idx=None):
    """
    Visualize a chunk with its labels.
    
    Args:
        chunk_data (dict): Chunk data from load_chunk()
        slice_idx (int): Slice index to visualize (default: middle)
    """
    volume = chunk_data['volume']
    labels = chunk_data.get('labels')
    
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    ct_slice = volume[slice_idx]
    
    if labels is None:
        # No labels - just show the slice
        plt.figure(figsize=(10, 10))
        plt.imshow(ct_slice, cmap='gray')
        plt.title(f"Chunk: {chunk_data['metadata']['chunk_id']}\nSlice {slice_idx}")
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        # Show slice and labels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original slice
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title('CT Scan Slice')
        axes[0].axis('off')
        
        # Labels
        axes[1].imshow(labels, cmap='hot')
        axes[1].set_title('Ground Truth Labels')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(ct_slice, cmap='gray', alpha=0.7)
        axes[2].imshow(labels, cmap='hot', alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f"Chunk: {chunk_data['metadata']['chunk_id']} | Slice {slice_idx}/{volume.shape[0]}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== Vesuvius Data Loader Demo ===\n")
    
    # This is a demo - you need to provide your actual data path
    print("To use this loader with real data:")
    print("1. Download Vesuvius Challenge dataset")
    print("2. Initialize loader: loader = VesuviusDataLoader('/path/to/dataset')")
    print("3. List chunks: chunks = loader.list_chunks()")
    print("4. Load chunk: data = loader.load_chunk(chunks[0])")
    print("5. Visualize: visualize_chunk_with_labels(data)")
    
    print("\nExample structure:")
    print("dataset/")
    print("  scroll1_chunk1/")
    print("    surface_volume/")
    print("      slice_0000.tif")
    print("      slice_0001.tif")
    print("      ...")
    print("    inklabels.png")
    print("    metadata.json")
