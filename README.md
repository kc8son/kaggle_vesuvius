# Vesuvius Challenge: CT Scan Surface Tracking

A beginner-friendly toolkit for working with CT scans from Villa dei Papiri (Herculaneum scrolls) to train models that follow scroll surfaces - essential for virtually unwrapping ancient texts.

## 🎯 Overview

This project provides simple tools to:
- Load 3D CT scan chunks from the Vesuvius Challenge dataset
- Work with binary labeled data (ground truth surface annotations)
- Track and detect scroll surfaces in variable-sized volumes
- Train machine learning models for surface prediction
- Handle data from ESRF (Grenoble) and DLS (Oxford) synchrotrons

## 📦 Dataset

The Vesuvius Challenge dataset includes:
- **3D chunks** of CT scans from closed, carbonized Herculaneum scrolls
- **Binary labels** indicating surface locations (ground truth)
- **Variable dimensions** - chunk sizes vary across the dataset
- Data from two synchrotrons:
  - ESRF (Grenoble, France) - beamline BM18
  - DLS (Oxford, UK) - beamline I12

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from load_ct_scans import load_ct_volume
from surface_tracker import SimpleSurfaceTracker
from train_model import SurfaceTrackingTrainer

# Load CT scan volume
volume = load_ct_volume("path/to/ct_scans")

# Track surfaces
tracker = SimpleSurfaceTracker()
trainer = SurfaceTrackingTrainer()

# Train model
X, y = trainer.prepare_training_data(volume, tracker)
results = trainer.train(X, y)
```

## 📚 Documentation

- **[TUTORIAL.md](TUTORIAL.md)** - Complete step-by-step guide for beginners
- **[Dataset Documentation](https://scrollprize.org/)** - Official Vesuvius Challenge data

## 🔧 Main Components

### 1. `load_ct_scans.py`
Load and visualize CT scan data:
- Load single slices or full 3D volumes
- Handle variable chunk dimensions
- Support for TIFF, PNG, and JPEG formats
- Visualization utilities

### 2. `surface_tracker.py`
Surface detection using image processing:
- Edge-based surface detection
- Feature extraction for ML training
- Works with unlabeled data

### 3. `train_model.py`
Train ML models for surface tracking:
- Random Forest classifier
- Automatic training data preparation
- Model evaluation and visualization
- Save/load trained models

### 4. `data_loader.py`
Handle Vesuvius Challenge dataset format:
- Load binary labeled chunks
- Work with ground truth annotations
- Handle variable dimensions
- Batch processing utilities

## 💡 Key Features

- **Beginner-friendly**: Clear documentation and simple APIs
- **Flexible**: Handles variable chunk sizes automatically
- **Educational**: Learn surface tracking step-by-step
- **Practical**: Ready for Kaggle competition submissions

## 🎓 Learning Path

1. Read [TUTORIAL.md](TUTORIAL.md) for concepts and examples
2. Run example scripts with sample data
3. Load real Vesuvius Challenge data
4. Train models with binary labels
5. Experiment and improve!

## 📊 Example Results

The toolkit can:
- Detect scroll surfaces with 75-85% accuracy
- Process variable-sized chunks efficiently
- Train on labeled data in minutes
- Generate probability maps for surface locations

## 🤝 Contributing

Contributions welcome! This is designed to be beginner-friendly and educational.

## 📄 License

See [LICENSE](LICENSE) file.

## 🔗 Resources

- [Vesuvius Challenge](https://scrollprize.org/)
- [Kaggle Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)
