import os
import pandas as pd
import numpy as np
from PIL import Image
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import autokeras as ak
import tensorflow as tf


def load_images_from_directory(base_path, algorithm_name, key_file="key.csv"):
    """
    Load images and labels from the algorithm-specific directory structure.

    Args:
        base_path: Path to the dataset (e.g., "datasets/sample")
        algorithm_name: Name of the algorithm subdirectory (e.g., "correlationPixel")
        key_file: Name of the key file (default "key.csv")

    Returns:
        tuple: (images_array, labels_array, filenames)
    """
    algorithm_path = os.path.join(base_path, algorithm_name)
    key_path = os.path.join(algorithm_path, key_file)

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found: {key_path}")

    # Load the key file
    key_df = pd.read_csv(key_path)
    print(f"Key file loaded: {len(key_df)} entries")

    images = []
    labels = []
    filenames = []

    for _, row in key_df.iterrows():
        filename = row['filename']
        label = row['label']

        # Load image
        image_path = os.path.join(algorithm_path, filename)
        if os.path.exists(image_path):
            try:
                # Load image and convert to RGB if needed
                img = Image.open(image_path)
                print(f"Loaded image {filename}: mode={img.mode}, size={img.size}")

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert to numpy array and normalize to [0,1]
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Ensure minimum size for CNN (resize if too small)
                if img_array.shape[0] < 32 or img_array.shape[1] < 32:
                    img = img.resize((32, 32), Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32) / 255.0

                images.append(img_array)
                labels.append(label)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")

    if len(images) == 0:
        raise ValueError("No images were successfully loaded")

    images_array = np.array(images)
    labels_array = np.array(labels)

    print(f"Final arrays - Images: {images_array.shape}, Labels: {labels_array.shape}")
    print(f"Image data range: [{images_array.min():.3f}, {images_array.max():.3f}]")
    print(f"Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")

    return images_array, labels_array, filenames


def autokeras_testing(base_path, algorithm_name, n_runs=5, test_size=0.2, max_trials=3, epochs=20):
    """
    Run AutoKeras image classification testing with multiple trials.

    Args:
        base_path: Path to the dataset directory
        algorithm_name: Name of the algorithm subdirectory
        n_runs: Number of evaluation runs
        test_size: Fraction of data to use for testing
        max_trials: Maximum number of trials for AutoKeras
        epochs: Number of epochs for training

    Returns:
        pd.DataFrame: DataFrame with columns ['model_name', 'run', 'accuracy', 'precision', 'recall', 'f1_score', 'train_time']
    """
    # Load images and labels
    print(f"Loading images from {base_path}/{algorithm_name}...")
    images, labels, filenames = load_images_from_directory(base_path, algorithm_name)

    if len(images) == 0:
        raise ValueError("No images loaded. Please check the directory structure and key file.")

    print(f"Loaded {len(images)} images with shape {images[0].shape}")
    print(f"Unique labels: {np.unique(labels)}")

    # Check if binary or multiclass
    unique_labels = np.unique(labels)
    is_binary = len(unique_labels) == 2
    num_classes = len(unique_labels)

    # Store all individual results
    results_list = []

    # Run multiple trials
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        # Split the data with different random state for each run
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=run, stratify=labels
        )

        print(f"Train set: {len(X_train)} images, Test set: {len(X_test)} images")
        print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        # Create AutoKeras classifier with more conservative settings
        clf = ak.ImageClassifier(
            max_trials=max_trials,
            overwrite=True,
            seed=run,
            directory=f'autokeras_tmp_{algorithm_name}_{run}'
        )

        try:
            # Time the training
            start_time = time.time()

            # Train the model with validation split
            clf.fit(
                X_train, y_train,
                epochs=epochs,
                validation_split=0.2,
                verbose=1
            )

            train_time = time.time() - start_time

            # Make predictions
            y_pred_proba = clf.predict(X_test)
            print(f"Prediction shape: {y_pred_proba.shape}")
            print(f"Prediction sample: {y_pred_proba[:5]}")

            # Convert predictions to labels
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                # Multi-class case
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # Binary case
                if is_binary:
                    y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
                else:
                    y_pred = y_pred_proba.flatten().astype(int)

            print(f"Predicted labels: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
            print(f"True labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            if is_binary:
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
            else:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            print(
                f"Run {run + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Time: {train_time:.2f}s")

        except Exception as e:
            print(f"Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()

            # Set default values for failed runs
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            train_time = 0.0

        # Store results as a row
        results_list.append({
            'model_name': f'AutoKeras_{algorithm_name}',
            'run': run + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time
        })

        # Clean up to free memory
        try:
            del clf
            tf.keras.backend.clear_session()
            # Clean up temporary directory
            import shutil
            temp_dir = f'autokeras_tmp_{algorithm_name}_{run}'
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Cleanup warning: {cleanup_error}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df


def get_available_algorithms(base_path):
    """
    Get list of available algorithm directories in the dataset path.

    Args:
        base_path: Path to the dataset directory

    Returns:
        list: List of algorithm directory names
    """
    if not os.path.exists(base_path):
        return []

    algorithms = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "key.csv")):
            algorithms.append(item)

    return algorithms