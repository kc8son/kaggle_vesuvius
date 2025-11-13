# Getting Started with Vesuvius Challenge Surface Tracking

## Quick Start (5 minutes)

### Installation
```bash
# Clone the repository
git clone https://github.com/kc8son/kaggle_vesuvius.git
cd kaggle_vesuvius

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Example
```bash
# Run the quick start example (no dataset needed)
python example_workflow.py
```

This will:
- Create sample CT scan data
- Train a surface tracking model
- Save the trained model
- Display accuracy metrics

Expected output: ~85% test accuracy in under 1 minute

## Next Steps

### 1. Explore with Sample Data
```python
# Interactive exploration
from load_ct_scans import create_sample_data, load_ct_volume, visualize_slice
from surface_tracker import SimpleSurfaceTracker

# Create and load sample data
create_sample_data("/tmp/my_scans", num_slices=20)
volume = load_ct_volume("/tmp/my_scans")

# Track surfaces
tracker = SimpleSurfaceTracker()
result = tracker.track_surface_in_slice(volume[10])

# Visualize (requires display)
# tracker.visualize_tracking(volume[10], result)
```

### 2. Work with Real Vesuvius Challenge Data

Download the dataset from Kaggle, then:

```python
from data_loader import VesuviusDataLoader, LabeledDataset
from train_model import SurfaceTrackingTrainer

# Load real dataset
loader = VesuviusDataLoader("/path/to/vesuvius/dataset")
chunks = loader.list_chunks()
print(f"Found {len(chunks)} chunks")

# Create training dataset
labeled_chunks = [c for c in chunks if loader.get_chunk_info(c)['has_labels']]
dataset = LabeledDataset(loader, labeled_chunks[:3])

# Train with ground truth labels
X, y = dataset.get_training_samples(num_samples_per_chunk=1000)
trainer = SurfaceTrackingTrainer()
results = trainer.train(X, y)

# Save model
trainer.save_model("vesuvius_model.pkl")
```

### 3. Use Advanced Papyrus Segmentation

```python
from papyrus_segmenter import PapyrusSegmenter

# Initialize with topology constraints
segmenter = PapyrusSegmenter(
    min_component_size=100,
    max_hole_size=50,
    merge_distance_threshold=5
)

# Load a chunk
chunk_data = loader.load_chunk(chunks[0])
ct_slice = chunk_data['volume'][chunk_data['volume'].shape[0] // 2]

# Get predictions from your model
prob_map = trainer.predict(ct_slice, stride=5)

# Segment with topology preservation
result = segmenter.segment_papyrus_sheet(
    ct_slice, 
    prob_map,
    prefer_recto=True,
    threshold=0.5
)

# Check quality
print(f"Components: {result['stats']['num_components']}")
print(f"Topology issues: {len(result['stats']['topology_issues'])}")
```

## File Guide

| File | Purpose | Use When |
|------|---------|----------|
| `load_ct_scans.py` | Load CT scan data | You need to read TIFF/PNG/JPEG scan files |
| `surface_tracker.py` | Detect surfaces unsupervised | You don't have labeled data |
| `data_loader.py` | Load Vesuvius dataset | You have the official competition dataset |
| `train_model.py` | Train ML models | You want to train a classifier |
| `papyrus_segmenter.py` | Advanced segmentation | You need production-quality results |
| `example_workflow.py` | Complete examples | You want to see everything working together |

## Common Workflows

### Workflow 1: No Data Available
```bash
python example_workflow.py
```
Uses sample data, trains a model, demonstrates all features.

### Workflow 2: Have Unlabeled CT Scans
```python
from load_ct_scans import load_ct_volume
from surface_tracker import SimpleSurfaceTracker
from train_model import SurfaceTrackingTrainer

volume = load_ct_volume("path/to/scans")
tracker = SimpleSurfaceTracker()
trainer = SurfaceTrackingTrainer()
X, y = trainer.prepare_training_data(volume, tracker)
results = trainer.train(X, y)
```

### Workflow 3: Have Vesuvius Challenge Dataset
```bash
python example_workflow.py /path/to/vesuvius/dataset
```
Loads real data, trains with ground truth labels, saves model.

## Performance Tips

- **Speed up training**: Reduce `samples_per_slice` (default 200 → 100)
- **Speed up prediction**: Increase `stride` (default 5 → 10)
- **Improve accuracy**: Use more training data (more chunks)
- **Better topology**: Tune `PapyrusSegmenter` parameters

## Troubleshooting

**Problem**: Out of memory
- **Solution**: Process fewer slices at once, reduce volume size in `load_ct_volume()`

**Problem**: Training is slow
- **Solution**: The model uses `n_jobs=-1` (all CPUs). Check CPU availability.

**Problem**: Poor predictions on real data
- **Solution**: Sample data is simplified. Adjust parameters:
  - `smoothing_sigma`: 1.0 → 3.0 for noisy data
  - `threshold_percentile`: 75 → 80 for clearer surfaces
  - `window_size`: 5 → 7 for more context

**Problem**: Can't visualize (no display)
- **Solution**: Comment out visualization calls, or save to file:
  ```python
  plt.savefig("output.png")
  ```

## Learning Path

1. ✅ **Start Here**: Run `example_workflow.py` to see it work
2. 📖 **Understand**: Read `TUTORIAL.md` for concepts
3. 🔬 **Experiment**: Modify parameters in examples
4. 📊 **Real Data**: Download and work with competition dataset
5. 🎯 **Optimize**: Tune for your specific scrolls
6. 🏆 **Compete**: Submit to Kaggle!

## Need Help?

- Read the detailed tutorial: `TUTORIAL.md`
- Check example code: `example_workflow.py`
- Run integration tests: `python test_integration.py`
- Review the README: `README.md`

## Dataset Structure Expected

```
vesuvius_dataset/
├── scroll1_chunk1/
│   ├── surface_volume/     # or just "volume"
│   │   ├── slice_0000.tif
│   │   ├── slice_0001.tif
│   │   └── ...
│   ├── inklabels.png       # Binary labels (optional)
│   └── metadata.json       # Optional metadata
├── scroll1_chunk2/
│   └── ...
```

Compatible with official Vesuvius Challenge dataset format from Kaggle.

## Key Capabilities

✅ Load CT scans (TIFF, PNG, JPEG)
✅ Handle variable chunk dimensions
✅ Work with or without labels
✅ Detect papyrus structure (recto/verso)
✅ Train Random Forest models
✅ Preserve topology (no mergers/holes)
✅ Evaluate segmentation quality
✅ Save/load trained models
✅ Generate predictions for submission

## Success Metrics

Expected performance on sample data:
- Training time: 10-60 seconds (depends on data size)
- Test accuracy: 75-85%
- Components detected: 1-5 per slice
- Topology issues: 0-2 minor issues

On real data, performance varies based on:
- Scroll condition (carbonization damage)
- CT scan quality (resolution, noise)
- Training data quantity and quality

Good luck with the Vesuvius Challenge! 🏺📜
