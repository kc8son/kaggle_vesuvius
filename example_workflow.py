"""
Example workflow for Vesuvius Challenge dataset.

This script demonstrates how to:
1. Load 3D chunks with binary labels
2. Work with variable-sized chunks
3. Train models using ground truth annotations
4. Make predictions on new data
"""

import numpy as np
from pathlib import Path

from data_loader import VesuviusDataLoader, LabeledDataset, visualize_chunk_with_labels
from train_model import SurfaceTrackingTrainer
from surface_tracker import SimpleSurfaceTracker


def example_workflow(data_root):
    """
    Complete example workflow with Vesuvius Challenge dataset.
    
    Args:
        data_root (str): Path to dataset root directory
    """
    print("="*60)
    print("VESUVIUS CHALLENGE - SURFACE TRACKING WORKFLOW")
    print("="*60)
    
    # Step 1: Initialize data loader
    print("\n[Step 1] Initializing data loader...")
    loader = VesuviusDataLoader(data_root)
    
    # Step 2: List available chunks
    print("\n[Step 2] Listing available chunks...")
    chunks = loader.list_chunks()
    print(f"Found {len(chunks)} chunks:")
    for chunk_id in chunks[:5]:  # Show first 5
        info = loader.get_chunk_info(chunk_id)
        print(f"  - {chunk_id}: shape={info['shape']}, has_labels={info['has_labels']}")
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more")
    
    if len(chunks) == 0:
        print("\n⚠️  No chunks found. Please check your data_root path.")
        return
    
    # Step 3: Load a sample chunk
    print(f"\n[Step 3] Loading chunk: {chunks[0]}...")
    chunk_data = loader.load_chunk(chunks[0], load_labels=True)
    print(f"  Volume shape: {chunk_data['volume'].shape}")
    print(f"  Has labels: {chunk_data['metadata']['has_labels']}")
    if chunk_data['labels'] is not None:
        print(f"  Labels shape: {chunk_data['labels'].shape}")
        print(f"  Surface pixels: {np.sum(chunk_data['labels'] > 0)} "
              f"({100*np.sum(chunk_data['labels'] > 0)/chunk_data['labels'].size:.2f}%)")
    
    # Step 4: Prepare labeled dataset
    print("\n[Step 4] Preparing labeled dataset...")
    # Use chunks with labels
    labeled_chunks = [c for c in chunks if loader.get_chunk_info(c)['has_labels']]
    
    if len(labeled_chunks) == 0:
        print("⚠️  No labeled chunks found. Training with labeled data not possible.")
        print("   Falling back to unsupervised surface detection...")
        
        # Use unsupervised approach
        tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)
        trainer = SurfaceTrackingTrainer(window_size=5)
        
        volume = chunk_data['volume']
        X, y = trainer.prepare_training_data(volume, tracker, samples_per_slice=200)
        
    else:
        print(f"Found {len(labeled_chunks)} chunks with labels")
        
        # Use up to 3 chunks for training (or all if fewer)
        train_chunks = labeled_chunks[:min(3, len(labeled_chunks))]
        dataset = LabeledDataset(loader, train_chunks)
        
        # Extract training samples
        X, y = dataset.get_training_samples(num_samples_per_chunk=1000, balance=True)
    
    # Step 5: Train model
    print("\n[Step 5] Training surface tracking model...")
    trainer = SurfaceTrackingTrainer(window_size=5)
    
    # Note: X, y are already prepared from Step 4
    results = trainer.train(X, y, test_size=0.2)
    
    print(f"\n📊 Training Results:")
    print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.3f}")
    
    # Step 6: Save model
    print("\n[Step 6] Saving trained model...")
    model_path = "vesuvius_surface_model.pkl"
    trainer.save_model(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Step 7: Make predictions on a test chunk
    print("\n[Step 7] Testing predictions...")
    test_slice = chunk_data['volume'][chunk_data['volume'].shape[0] // 2]
    prob_map = trainer.predict(test_slice, stride=10)
    
    print(f"  Prediction map shape: {prob_map.shape}")
    print(f"  Predicted surface area: {np.sum(prob_map > 0.5)} pixels")
    print(f"  Confidence range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Summary
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE")
    print("="*60)
    print(f"Model trained on {X.shape[0]} samples")
    print(f"Test accuracy: {results['test_accuracy']:.1%}")
    print(f"Model saved to: {model_path}")
    print("\nNext steps:")
    print("  - Visualize results: visualize_chunk_with_labels(chunk_data)")
    print("  - Test on more chunks")
    print("  - Tune hyperparameters")
    print("  - Submit to Kaggle competition!")


def quick_start_example():
    """
    Quick start example with sample data (no dataset required).
    """
    print("="*60)
    print("QUICK START - SAMPLE DATA")
    print("="*60)
    
    from load_ct_scans import create_sample_data, load_ct_volume
    
    # Create sample data
    print("\n[1] Creating sample CT scan data...")
    sample_folder = "/tmp/vesuvius_sample"
    create_sample_data(sample_folder, num_slices=15, size=(256, 256))
    
    # Load volume
    print("\n[2] Loading volume...")
    volume = load_ct_volume(sample_folder)
    
    # Track surfaces
    print("\n[3] Detecting surfaces...")
    tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)
    
    # Train model
    print("\n[4] Training model...")
    trainer = SurfaceTrackingTrainer(window_size=5)
    X, y = trainer.prepare_training_data(volume, tracker, samples_per_slice=150)
    results = trainer.train(X, y, test_size=0.2)
    
    # Save model
    print("\n[5] Saving model...")
    trainer.save_model("sample_model.pkl")
    
    # Test prediction
    print("\n[6] Testing prediction...")
    test_slice = volume[-1]
    prob_map = trainer.predict(test_slice, stride=8)
    
    print("\n" + "="*60)
    print("✅ QUICK START COMPLETE")
    print("="*60)
    print(f"Test accuracy: {results['test_accuracy']:.1%}")
    print("Model saved to: sample_model.pkl")
    print("\nTo work with real data, use: example_workflow('/path/to/dataset')")


if __name__ == "__main__":
    import sys
    
    print("\n🎯 VESUVIUS CHALLENGE - SURFACE TRACKING\n")
    
    # Check if data path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Using dataset at: {data_path}\n")
        example_workflow(data_path)
    else:
        print("No dataset path provided. Running quick start with sample data...\n")
        print("To use real data, run:")
        print("  python example_workflow.py /path/to/vesuvius/dataset\n")
        quick_start_example()
