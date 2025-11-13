# Vesuvius Challenge: CT Scan Surface Tracking Tutorial

This tutorial will guide you through working with CT scans from Villa dei Papiri to train a model for scroll surface tracking - a crucial step in virtually unwrapping ancient scrolls.

## 📋 What You'll Learn

1. How to load and visualize CT scan data
2. How to detect surfaces in 3D CT volumes
3. How to train a simple machine learning model for surface tracking
4. How to use the trained model to predict surface locations

## 🚀 Quick Start

### Prerequisites

First, install the required packages:

```bash
pip install -r requirements.txt
```

### Understanding the Problem

Ancient scrolls (like those from Villa dei Papiri) are often too fragile to unroll physically. Instead, we use CT scans to capture 3D images of the rolled scroll. To read the text, we need to:

1. **Identify the surface** of the papyrus layers within the scroll
2. **Track the surface** as it spirals through the 3D volume
3. **Virtually unwrap** the surface to reveal the text

This project focuses on **step 1 and 2**: identifying and tracking surfaces.

#### Papyrus Structure

Each papyrus sheet has **two layers**:
- **Recto**: The inner surface with horizontal fibers (faces the umbilicus/scroll center)
- **Verso**: The outer surface with vertical fibers

**Goal**: Ideally detect the recto surface, but detecting the overall sheet position (encompassing both recto and verso) is acceptable for virtual unwrapping.

**Critical Constraint**: Avoid topological errors:
- ❌ No artificial mergers between different sheets
- ❌ No holes that split a single sheet into disconnected parts

The scrolls were carbonized in a volcanic eruption, so sheets can be damaged and frayed.

## 📚 Step-by-Step Guide

### Step 1: Load CT Scan Data

The first step is loading CT scan slices. Each slice is a 2D image representing one "level" of the 3D scan.

```python
from load_ct_scans import load_ct_slice, load_ct_volume, visualize_slice

# Load a single slice
ct_slice = load_ct_slice("path/to/slice_0001.tif")

# Visualize it
visualize_slice(ct_slice, title="My First CT Slice")

# Load multiple slices as a 3D volume
volume = load_ct_volume(
    folder_path="path/to/ct_scans",
    start_slice=0,
    end_slice=50,
    step=1  # Use every slice
)

print(f"Volume shape: {volume.shape}")  # (num_slices, height, width)
```

**What's happening?**
- `load_ct_slice()`: Loads a single 2D image
- `load_ct_volume()`: Loads multiple slices into a 3D array
- `visualize_slice()`: Shows the image with proper scaling

### Step 2: Detect Surfaces

Now we'll detect where surfaces (papyrus layers) are located in the CT scan.

```python
from surface_tracker import SimpleSurfaceTracker

# Initialize the tracker
tracker = SimpleSurfaceTracker(
    smoothing_sigma=2.0,        # Smooth the image to reduce noise
    threshold_percentile=75     # Keep top 25% of edge strengths
)

# Track surface in a single slice
result = tracker.track_surface_in_slice(ct_slice, num_points=100)

print(f"Found {len(result['surface_points'])} surface points")

# Visualize the results
tracker.visualize_tracking(ct_slice, result, title="Surface Detection")
```

**What's happening?**
- The tracker uses **edge detection** to find boundaries between different materials
- Papyrus appears different from air in CT scans, creating edges
- The tracker identifies the strongest edges as potential surface points

### Step 3: Extract Features

To train a machine learning model, we need to convert surface points into numerical features.

```python
from surface_tracker import SurfaceFeatureExtractor

# Initialize feature extractor
feature_extractor = SurfaceFeatureExtractor(window_size=5)

# Extract features for surface points
features = feature_extractor.extract_features_from_points(
    ct_slice, 
    result['surface_points']
)

print(f"Feature matrix shape: {features.shape}")
# Each point has 6 features: mean, std, median, min, max, range
```

**What's happening?**
- For each point, we look at a small window around it (e.g., 5x5 pixels)
- We calculate statistics: mean intensity, standard deviation, etc.
- These features help the model learn what "surface" looks like

### Step 4: Train a Model

Now we'll train a machine learning model to automatically identify surfaces.

```python
from train_model import SurfaceTrackingTrainer

# Initialize trainer
trainer = SurfaceTrackingTrainer(window_size=5)

# Prepare training data from a volume
X, y = trainer.prepare_training_data(
    volume, 
    tracker, 
    samples_per_slice=200
)

# Train the model
results = trainer.train(X, y, test_size=0.2)

print(f"Test accuracy: {results['test_accuracy']:.3f}")

# Save the trained model
trainer.save_model("my_surface_model.pkl")
```

**What's happening?**
- The trainer automatically labels surface points (y=1) and background points (y=0)
- A Random Forest model learns patterns that distinguish surfaces from background
- We evaluate on test data to see how well it generalizes

### Step 5: Make Predictions

Use your trained model to predict surfaces in new CT scans.

```python
# Load a new CT slice
new_slice = load_ct_slice("path/to/new_slice.tif")

# Predict surface locations
prob_map = trainer.predict(new_slice, stride=5)

# Visualize predictions
from train_model import visualize_predictions
visualize_predictions(new_slice, prob_map)
```

**What's happening?**
- The model evaluates every point in the image
- It outputs a probability (0-1) that each point is on a surface
- High probability areas (red) indicate likely surface locations

### Step 6: Papyrus-Specific Segmentation (Advanced)

For production-quality results, use the papyrus-aware segmenter that handles topology:

```python
from papyrus_segmenter import PapyrusSegmenter

# Initialize segmenter with topology constraints
segmenter = PapyrusSegmenter(
    min_component_size=100,      # Remove noise
    max_hole_size=50,            # Fill small holes
    merge_distance_threshold=5   # Prevent sheet mergers
)

# Get probability map from your trained model
prob_map = trainer.predict(new_slice, stride=5)

# Segment with topology preservation
result = segmenter.segment_papyrus_sheet(
    new_slice, 
    prob_map, 
    prefer_recto=True,  # Prefer horizontal fibers (recto surface)
    threshold=0.5
)

# Visualize with topology information
segmenter.visualize_segmentation(new_slice, result)

# Check results
print(f"Components found: {result['stats']['num_components']}")
print(f"Recto pixels: {result['recto_pixels']}")
print(f"Verso pixels: {result['verso_pixels']}")
if result['stats']['topology_issues']:
    print("⚠️  Topology issues detected!")
```

**What's happening?**
- Detects fiber orientation (horizontal=recto, vertical=verso)
- Applies morphological operations to preserve topology
- Prevents artificial mergers between sheets
- Fills small holes but preserves genuine gaps
- Reports topology quality metrics

## 🎯 Complete Example

Here's a complete workflow from start to finish:

```python
from load_ct_scans import create_sample_data, load_ct_volume
from surface_tracker import SimpleSurfaceTracker
from train_model import SurfaceTrackingTrainer, visualize_predictions

# 1. Create sample data (for testing)
create_sample_data("/tmp/my_scans", num_slices=20)

# 2. Load the data
volume = load_ct_volume("/tmp/my_scans")

# 3. Initialize tracker
tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)

# 4. Train model
trainer = SurfaceTrackingTrainer(window_size=5)
X, y = trainer.prepare_training_data(volume, tracker, samples_per_slice=200)
results = trainer.train(X, y)

# 5. Test on a new slice
test_slice = volume[-1]
prob_map = trainer.predict(test_slice, stride=5)
visualize_predictions(test_slice, prob_map)

# 6. Save model
trainer.save_model("surface_model.pkl")
```

## 🔧 Working with Real Data

When you have real CT scan data from the Vesuvius Challenge:

### Option 1: Using Labeled Data (Recommended)

The Vesuvius Challenge provides 3D chunks with binary labels (ground truth):

```python
from data_loader import VesuviusDataLoader, LabeledDataset

# 1. Initialize data loader
loader = VesuviusDataLoader("/path/to/vesuvius/dataset")

# 2. List available chunks
chunks = loader.list_chunks()
print(f"Available chunks: {chunks}")

# 3. Load a chunk with labels
chunk_data = loader.load_chunk(chunks[0], load_labels=True)
print(f"Volume shape: {chunk_data['volume'].shape}")
print(f"Has labels: {chunk_data['metadata']['has_labels']}")

# 4. Create labeled dataset
labeled_chunks = [c for c in chunks if loader.get_chunk_info(c)['has_labels']]
dataset = LabeledDataset(loader, labeled_chunks[:3])  # Use first 3 chunks

# 5. Train with ground truth labels
X, y = dataset.get_training_samples(num_samples_per_chunk=1000, balance=True)
trainer = SurfaceTrackingTrainer(window_size=5)
results = trainer.train(X, y)
```

**Dataset Structure:**
```
vesuvius_dataset/
├── scroll1_chunk1/
│   ├── surface_volume/
│   │   ├── slice_0000.tif
│   │   ├── slice_0001.tif
│   │   └── ...
│   ├── inklabels.png      # Binary ground truth
│   └── metadata.json
├── scroll1_chunk2/
│   └── ...
```

### Option 2: Using Unlabeled Data

If you only have CT scans without labels:

```python
from load_ct_scans import load_ct_volume
from surface_tracker import SimpleSurfaceTracker

# Load volume
volume = load_ct_volume("path/to/ct_scans", start_slice=0, end_slice=100)

# Use unsupervised surface detection
tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)
trainer = SurfaceTrackingTrainer(window_size=5)
X, y = trainer.prepare_training_data(volume, tracker, samples_per_slice=200)
results = trainer.train(X, y)
```

### Working with New Labeled Data

The Vesuvius team releases new labeled data throughout the competition. These new labels may be less curated:

```python
# Evaluate data quality before training
chunk_data = loader.load_chunk("new_chunk_id", load_labels=True)

# Visualize to assess quality
from data_loader import visualize_chunk_with_labels
visualize_chunk_with_labels(chunk_data)

# Check label statistics
labels = chunk_data['labels']
label_density = labels.sum() / labels.size
print(f"Label density: {label_density:.2%}")

# If quality looks good, include in training
if label_density > 0.01 and label_density < 0.5:  # Reasonable range
    # Add to training set
    dataset = LabeledDataset(loader, ["new_chunk_id"])
    X_new, y_new = dataset.get_training_samples()
    # Combine with existing data or train separately
```

### Parameter Tuning
   - `smoothing_sigma`: Increase if data is noisy
   - `threshold_percentile`: Adjust to capture more/fewer edges
   - `window_size`: Larger windows capture more context
   - `samples_per_slice`: More samples = more training data

4. **Train with your data**:
   ```python
   tracker = SimpleSurfaceTracker(smoothing_sigma=3.0, threshold_percentile=80)
   trainer = SurfaceTrackingTrainer(window_size=7)
   X, y = trainer.prepare_training_data(volume, tracker, samples_per_slice=500)
   results = trainer.train(X, y)
   ```

## 💡 Tips for Beginners

### Understanding the Approach

This implementation uses a **two-stage approach**:

1. **Rule-based surface detection** (SimpleSurfaceTracker)
   - Uses traditional image processing (edge detection)
   - Fast and interpretable
   - Creates training labels automatically

2. **Machine learning model** (SurfaceTrackingTrainer)
   - Learns from the rule-based detector
   - Can generalize to new patterns
   - More robust to noise

### Common Issues and Solutions

**Issue**: "No image files found"
- **Solution**: Check that your folder contains .tif, .png, or .jpg files

**Issue**: Too few/many surface points detected
- **Solution**: Adjust `threshold_percentile` (lower = more points, higher = fewer points)

**Issue**: Model accuracy is low
- **Solution**: 
  - Collect more training data (more slices)
  - Increase `samples_per_slice`
  - Adjust `window_size` for feature extraction
  - Try different `smoothing_sigma` values
  - Use labeled data instead of unsupervised detection

**Issue**: Predictions are too slow
- **Solution**: Increase `stride` parameter in `predict()` (trades speed for resolution)

**Issue**: Artificial sheet mergers in segmentation
- **Solution**:
  - Increase `merge_distance_threshold` in PapyrusSegmenter
  - Lower the probability threshold
  - Apply morphological opening

**Issue**: Holes splitting single sheets
- **Solution**:
  - Increase `max_hole_size` in PapyrusSegmenter
  - Apply binary hole filling
  - Check for noise in original CT data

**Issue**: New labeled data quality is poor
- **Solution**:
  - Visualize before training: `visualize_chunk_with_labels(chunk_data)`
  - Check label density (should be 1-30% typically)
  - Train separate models and ensemble
  - Use data augmentation to compensate

### Topology Quality Checklist

Before submitting predictions, verify:
- ✅ Each papyrus sheet is a single connected component
- ✅ No artificial bridges connecting separate sheets
- ✅ No large holes within sheets (small damage is okay)
- ✅ Smooth boundaries (no excessive jaggedness)
- ✅ Reasonable coverage (not too sparse or dense)

### Next Steps

After mastering surface tracking, you can:

1. **3D surface reconstruction**: Connect surface points across slices to build a 3D model
2. **Surface refinement**: Use more advanced algorithms (active contours, level sets)
3. **Text detection**: Once surfaces are tracked, look for ink patterns on them
4. **Virtual unwrapping**: Flatten the 3D surface to reveal the text

## 📖 Additional Resources

- [Vesuvius Challenge](https://scrollprize.org/): Official challenge website
- [Digital Unwrapping Tutorial](https://www.youtube.com/watch?v=yHbpVcGD06U): Video on the full pipeline
- [Kaggle Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection): Competition page
- [CT Scan Basics](https://en.wikipedia.org/wiki/CT_scan): Understanding CT imaging
- [Edge Detection](https://en.wikipedia.org/wiki/Edge_detection): Theory behind surface detection
- [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest): The ML model we use
- [Papyrus Structure](https://en.wikipedia.org/wiki/Papyrus): Understanding papyrus composition

## 🤝 Contributing

This is a beginner-friendly project. Ideas for improvement:

- Add more feature types (texture, gradients, etc.)
- Implement 3D tracking across slices
- Add more visualization options
- Support for different CT scan formats
- Pre-trained models for common scroll types

## ❓ FAQ

**Q: Do I need a GPU?**
A: No, this implementation runs on CPU. It's designed to be accessible.

**Q: How much data do I need?**
A: Start with 20-50 slices for training. More is better, but not required.

**Q: Can I use this for other 3D medical imaging?**
A: Yes! The techniques work for any 3D scan where you need to track surfaces.

**Q: What if my CT scans have different formats?**
A: The code supports TIFF, PNG, and JPEG. For other formats, you may need to convert them first.

**Q: What's the difference between recto and verso?**
A: Recto has horizontal fibers (inner surface), verso has vertical fibers (outer surface). Recto is preferred but detecting both is acceptable.

**Q: How do I avoid topological errors?**
A: Use `PapyrusSegmenter` with appropriate parameters. It automatically checks for mergers and holes.

**Q: Should I use all the new labeled data released during competition?**
A: Not necessarily. Visualize each chunk first, check quality, and use your judgment. Less curated data may need filtering.

**Q: What accuracy should I expect?**
A: With labeled data: 75-90% accuracy. With unsupervised: 60-80%. Topology correctness is more important than raw accuracy.

## 📝 Summary

You now know how to:
- ✅ Load CT scan data
- ✅ Detect surfaces using edge detection
- ✅ Extract features from surface points
- ✅ Train a machine learning model
- ✅ Make predictions on new data

This is a solid foundation for the Vesuvius Challenge. Good luck with virtually unwrapping ancient scrolls! 🎯
