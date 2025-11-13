"""
Train a simple model for scroll surface tracking.

This script demonstrates how to train a basic model to predict surface locations
in CT scan data. It's designed to be beginner-friendly and educational.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

from load_ct_scans import load_ct_volume, create_sample_data
from surface_tracker import SimpleSurfaceTracker, SurfaceFeatureExtractor


class SurfaceTrackingTrainer:
    """
    Trainer for surface tracking model.
    
    This uses a Random Forest classifier to learn which regions contain
    surface pixels vs. background pixels.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize the trainer.
        
        Args:
            window_size (int): Size of local window for feature extraction
        """
        self.feature_extractor = SurfaceFeatureExtractor(window_size=window_size)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def prepare_training_data(self, volume, surface_tracker, samples_per_slice=200):
        """
        Prepare training data from a CT volume.
        
        Args:
            volume (numpy.ndarray): 3D CT scan volume
            surface_tracker (SimpleSurfaceTracker): Surface tracker for finding surface points
            samples_per_slice (int): Number of samples per slice
            
        Returns:
            tuple: (X_features, y_labels)
        """
        print(f"Preparing training data from {volume.shape[0]} slices...")
        
        X_list = []
        y_list = []
        
        for i in range(volume.shape[0]):
            ct_slice = volume[i]
            
            # Track surface points (positive examples)
            result = surface_tracker.track_surface_in_slice(ct_slice, num_points=samples_per_slice // 2)
            surface_points = result['surface_points']
            
            if len(surface_points) > 0:
                # Extract features for surface points
                surface_features = self.feature_extractor.extract_features_from_points(
                    ct_slice, surface_points
                )
                X_list.append(surface_features)
                y_list.append(np.ones(len(surface_points)))
            
            # Sample random background points (negative examples)
            num_background = samples_per_slice // 2
            bg_y = np.random.randint(0, ct_slice.shape[0], num_background)
            bg_x = np.random.randint(0, ct_slice.shape[1], num_background)
            background_points = np.column_stack([bg_y, bg_x])
            
            # Filter out points that are too close to surface points
            if len(surface_points) > 0:
                # Simple filtering: keep background points far from any surface point
                valid_bg = []
                for bg_point in background_points:
                    min_dist = np.min(np.sum((surface_points - bg_point)**2, axis=1))
                    if min_dist > 100:  # Distance threshold
                        valid_bg.append(bg_point)
                
                if len(valid_bg) > 0:
                    background_points = np.array(valid_bg)
                else:
                    background_points = background_points  # Use all if none valid
            
            # Extract features for background points
            background_features = self.feature_extractor.extract_features_from_points(
                ct_slice, background_points
            )
            X_list.append(background_features)
            y_list.append(np.zeros(len(background_points)))
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{volume.shape[0]} slices")
        
        # Combine all data
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        print(f"✓ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Surface samples: {np.sum(y == 1)}")
        print(f"  Background samples: {np.sum(y == 0)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train the surface tracking model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Labels (1 for surface, 0 for background)
            test_size (float): Fraction of data to use for testing
            
        Returns:
            dict: Training results including accuracy and predictions
        """
        print("\n--- Training Model ---")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        print("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        print("\n--- Evaluation ---")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred, 
                                   target_names=['Background', 'Surface']))
        
        # Feature importance
        feature_names = ['Mean', 'Std', 'Median', 'Min', 'Max', 'Range']
        importances = self.model.feature_importances_
        print("\nFeature Importances:")
        for name, imp in zip(feature_names, importances):
            print(f"  {name}: {imp:.3f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importances': importances,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': test_pred
        }
    
    def predict(self, ct_slice, stride=10):
        """
        Predict surface locations in a CT slice.
        
        Args:
            ct_slice (numpy.ndarray): CT scan slice
            stride (int): Stride for sampling points (smaller = more points, slower)
            
        Returns:
            numpy.ndarray: Probability map for surface locations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create a grid of points to evaluate
        y_coords = np.arange(0, ct_slice.shape[0], stride)
        x_coords = np.arange(0, ct_slice.shape[1], stride)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        points = np.column_stack([yy.ravel(), xx.ravel()])
        
        # Extract features for all points
        features = self.feature_extractor.extract_features_from_points(ct_slice, points)
        
        # Predict probabilities
        probs = self.model.predict_proba(features)[:, 1]  # Probability of surface class
        
        # Reshape to grid
        prob_map = np.zeros(ct_slice.shape)
        for i, (y, x) in enumerate(points):
            prob_map[y, x] = probs[i]
        
        return prob_map
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'window_size': self.feature_extractor.window_size
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_extractor = SurfaceFeatureExtractor(window_size=data['window_size'])
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")


def visualize_predictions(ct_slice, prob_map, title="Surface Predictions"):
    """
    Visualize model predictions on a CT slice.
    
    Args:
        ct_slice (numpy.ndarray): Original CT slice
        prob_map (numpy.ndarray): Predicted probability map
        title (str): Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original slice
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('Original CT Slice')
    axes[0].axis('off')
    
    # Probability map
    im = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Surface Probability Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(ct_slice, cmap='gray', alpha=0.7)
    axes[2].imshow(prob_map, cmap='hot', alpha=0.3, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Surface Tracking Training Demo ===\n")
    
    # 1. Create sample data
    sample_folder = "/tmp/sample_ct_scans"
    print("Step 1: Creating sample data...")
    create_sample_data(sample_folder, num_slices=20)
    
    # 2. Load CT volume
    print("\nStep 2: Loading CT volume...")
    volume = load_ct_volume(sample_folder, start_slice=0, end_slice=15)
    
    # 3. Initialize surface tracker
    print("\nStep 3: Initializing surface tracker...")
    surface_tracker = SimpleSurfaceTracker(smoothing_sigma=2.0, threshold_percentile=75)
    
    # 4. Prepare training data
    print("\nStep 4: Preparing training data...")
    trainer = SurfaceTrackingTrainer(window_size=5)
    X, y = trainer.prepare_training_data(volume, surface_tracker, samples_per_slice=200)
    
    # 5. Train model
    print("\nStep 5: Training model...")
    results = trainer.train(X, y, test_size=0.2)
    
    # 6. Test prediction on new slice
    print("\nStep 6: Testing prediction on a new slice...")
    test_slice = volume[-1]  # Use last slice as test
    prob_map = trainer.predict(test_slice, stride=5)
    print(f"Prediction map shape: {prob_map.shape}")
    print(f"Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # 7. Save model
    model_path = "/tmp/surface_tracking_model.pkl"
    trainer.save_model(model_path)
    
    print("\n" + "="*50)
    print("✓ Training complete!")
    print("="*50)
    print(f"\nModel saved to: {model_path}")
    print(f"Test accuracy: {results['test_accuracy']:.3f}")
    print("\nUse visualize_predictions() to see the results!")
