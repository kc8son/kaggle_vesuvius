"""
Integration test for the entire pipeline.
Tests all modules working together.
"""

def test_basic_workflow():
    """Test basic workflow with sample data."""
    print("="*60)
    print("INTEGRATION TEST: Basic Workflow")
    print("="*60)
    
    # Test 1: Load CT scans
    print("\n[Test 1] Loading CT scans...")
    from load_ct_scans import create_sample_data, load_ct_volume, load_ct_slice
    
    sample_folder = "/tmp/integration_test"
    create_sample_data(sample_folder, num_slices=10, size=(128, 128))
    volume = load_ct_volume(sample_folder)
    assert volume.shape == (10, 128, 128), f"Expected shape (10, 128, 128), got {volume.shape}"
    print("✓ CT scan loading works")
    
    # Test 2: Surface tracking
    print("\n[Test 2] Surface tracking...")
    from surface_tracker import SimpleSurfaceTracker, SurfaceFeatureExtractor
    
    tracker = SimpleSurfaceTracker(smoothing_sigma=1.0, threshold_percentile=70)
    result = tracker.track_surface_in_slice(volume[0], num_points=50)
    assert 'edges' in result, "Missing edges in result"
    assert 'surface_points' in result, "Missing surface_points in result"
    print(f"✓ Surface tracking works ({len(result['surface_points'])} points found)")
    
    # Test 3: Feature extraction
    print("\n[Test 3] Feature extraction...")
    extractor = SurfaceFeatureExtractor(window_size=3)
    if len(result['surface_points']) > 0:
        features = extractor.extract_features_from_points(volume[0], result['surface_points'])
        assert features.shape[1] == 6, f"Expected 6 features, got {features.shape[1]}"
        print(f"✓ Feature extraction works ({features.shape[0]} samples, {features.shape[1]} features)")
    
    # Test 4: Model training
    print("\n[Test 4] Model training...")
    from train_model import SurfaceTrackingTrainer
    
    trainer = SurfaceTrackingTrainer(window_size=3)
    X, y = trainer.prepare_training_data(volume[:8], tracker, samples_per_slice=50)
    assert len(X) > 0, "No training samples generated"
    assert len(X) == len(y), "Mismatch between features and labels"
    
    results = trainer.train(X, y, test_size=0.2)
    assert results['test_accuracy'] > 0.5, f"Accuracy too low: {results['test_accuracy']}"
    print(f"✓ Model training works (accuracy: {results['test_accuracy']:.2%})")
    
    # Test 5: Prediction
    print("\n[Test 5] Making predictions...")
    test_slice = volume[-1]
    prob_map = trainer.predict(test_slice, stride=10)
    assert prob_map.shape == test_slice.shape, "Probability map shape mismatch"
    assert prob_map.min() >= 0 and prob_map.max() <= 1, "Probabilities out of range"
    print(f"✓ Prediction works (prob range: [{prob_map.min():.3f}, {prob_map.max():.3f}])")
    
    # Test 6: Papyrus segmentation
    print("\n[Test 6] Papyrus segmentation...")
    from papyrus_segmenter import PapyrusSegmenter, evaluate_segmentation_quality
    
    segmenter = PapyrusSegmenter(min_component_size=20, max_hole_size=10)
    seg_result = segmenter.segment_papyrus_sheet(test_slice, prob_map, threshold=0.3)
    assert 'segmentation' in seg_result, "Missing segmentation in result"
    assert 'stats' in seg_result, "Missing stats in result"
    print(f"✓ Papyrus segmentation works ({seg_result['stats']['num_components']} components)")
    
    # Test 7: Save/load model
    print("\n[Test 7] Model persistence...")
    import os
    model_path = "/tmp/test_model.pkl"
    trainer.save_model(model_path)
    assert os.path.exists(model_path), "Model file not created"
    
    trainer2 = SurfaceTrackingTrainer(window_size=3)
    trainer2.load_model(model_path)
    assert trainer2.is_trained, "Loaded model not marked as trained"
    print("✓ Model save/load works")
    
    print("\n" + "="*60)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("="*60)
    return True


def test_data_loader_structure():
    """Test data loader can handle expected directory structures."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Data Loader")
    print("="*60)
    
    from data_loader import VesuviusDataLoader
    import os
    from pathlib import Path
    
    # Create mock dataset structure
    test_root = Path("/tmp/mock_vesuvius_dataset")
    chunk1_path = test_root / "scroll1_chunk1" / "volume"
    chunk1_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock slices
    from load_ct_scans import create_sample_data
    create_sample_data(str(chunk1_path), num_slices=5, size=(64, 64))
    
    # Create mock label
    import numpy as np
    from PIL import Image
    label = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8) * 255
    Image.fromarray(label).save(test_root / "scroll1_chunk1" / "inklabels.png")
    
    # Test loader
    print("\n[Test 1] Initialize loader...")
    loader = VesuviusDataLoader(test_root)
    print("✓ Loader initialized")
    
    print("\n[Test 2] List chunks...")
    chunks = loader.list_chunks()
    assert len(chunks) > 0, "No chunks found"
    print(f"✓ Found {len(chunks)} chunk(s)")
    
    print("\n[Test 3] Get chunk info...")
    info = loader.get_chunk_info(chunks[0])
    assert info['has_labels'], "Labels not detected"
    print(f"✓ Chunk info: shape={info['shape']}, has_labels={info['has_labels']}")
    
    print("\n[Test 4] Load chunk with labels...")
    chunk_data = loader.load_chunk(chunks[0], load_labels=True)
    assert chunk_data['volume'] is not None, "Volume not loaded"
    assert chunk_data['labels'] is not None, "Labels not loaded"
    print(f"✓ Loaded chunk: volume={chunk_data['volume'].shape}, labels={chunk_data['labels'].shape}")
    
    print("\n" + "="*60)
    print("✅ DATA LOADER TESTS PASSED")
    print("="*60)
    return True


if __name__ == "__main__":
    import sys
    
    try:
        # Run tests
        test_basic_workflow()
        test_data_loader_structure()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe toolkit is ready to use:")
        print("  - Load CT scans: load_ct_scans.py")
        print("  - Track surfaces: surface_tracker.py")
        print("  - Train models: train_model.py")
        print("  - Work with dataset: data_loader.py")
        print("  - Papyrus segmentation: papyrus_segmenter.py")
        print("  - Complete example: example_workflow.py")
        print("\nRead TUTORIAL.md for detailed instructions!")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
