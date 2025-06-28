import json
import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import autokeras as ak

def dataloader(path):
    """
    Load the images based on the key.csv file.
    Creating a numpy array of images and a list of labels.
    :return: (X, y) where X is numpy array of shape (num_samples, height, width, channels)
    """
    # load the key.csv file
    df = pd.read_csv(os.path.join(path, "key.csv"))

    x = []
    y = []

    for index, row in df.iterrows():
        filename = row['filename'].replace('.png', '.json')
        image_path = os.path.join(path, filename)

        # Load the JSON data
        with open(image_path, 'r') as f:
            image_data = json.load(f)

        # Convert to proper numpy array
        if isinstance(image_data, list):
            # If it's a flat list of RGB values, reshape appropriately
            image_array = np.array(image_data)

            # Reshape to proper image dimensions (you'll need to know your image dimensions)
            # For example, if it's a 28x28 RGB image:
            # image_array = image_array.reshape(28, 28, 3)

        else:
            # If it's already structured, convert directly
            image_array = np.array(image_data)

        # Ensure proper data type and range
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0  # Normalize to [0,1]

        x.append(image_array)
        y.append(row['label'])

    # Convert to numpy arrays
    X = np.array(x)  # Shape should be (num_samples, height, width, channels)
    y = np.array(y)  # Shape should be (num_samples,)

    return X, y


def autokeras_runner(path, n=1, epochs=10, batch_size=None, validation_split=None,
                     validation_data=None, max_trials=1, directory='auto_keras',
                     algorithm_name='image_classifier', overwrite=True, seed=None,
                     max_model_size=None, tuner='random', **kwargs):
    """
    Load the dataset from the given path and run AutoKeras with configurable parameters.
    """
    # get the data
    x, y = dataloader(path)

    results_list = []

    # Set default batch_size if not provided
    if batch_size is None:
        batch_size = min(32, len(x) // 2) if len(x) < 64 else 32

    # Set default validation if not provided
    if validation_split is None and validation_data is None:
        if len(x) < 64:
            validation_data = (x, y)
        else:
            validation_split = 0.2

    for run in range(n):
        start_time = time.time()

        clf = ak.ImageClassifier(
            max_trials=max_trials,
            directory=directory,
            project_name=f"{algorithm_name}_run_{run}",
            overwrite=overwrite,
            seed=seed,
            max_model_size=max_model_size,
            tuner=tuner
        )

        fit_kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            **kwargs
        }

        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
        elif validation_split is not None:
            fit_kwargs['validation_split'] = validation_split

        clf.fit(x, y, **fit_kwargs)
        train_time = time.time() - start_time

        model = clf.export_model()
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)

        # calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        results_list.append({
            'model_name': 'autokeras_cnn',
            'run': run + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time
        })

    return results_list


def get_algorithm_dir(path):
    algorithm_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    # for each of those algorithm dir check if there is a key.csv file if not remove it from the list
    algorithm_dirs = [d for d in algorithm_dirs if os.path.isfile(os.path.join(path, d, "key.csv"))]

    return algorithm_dirs


def autokeras_cnn_pipeline(path, output_folder, all_result_df, all_result_df_path=None, **kwargs):
    """
    Run the autokeras CNN pipeline on the dataset.
    The dataset should be a CSV file with a target column.
    """
    # get the list of all algorithm in the dataset folder
    algorithms = get_algorithm_dir(path)

    # for each algorithm run the autokeras CNN pipeline
    for algorithm in algorithms:
        # get the path to the algorithm folder
        algorithm_path = os.path.join(path, algorithm)

        # check if the key.csv file exists
        result = autokeras_runner(algorithm_path, directory=output_folder, algorithm_name=algorithm, **kwargs)
        # convert the results to a DataFrame
        result_df = pd.DataFrame(result)
        # merge the results with the main DataFrame
        all_result_df = pd.concat([all_result_df, result_df], ignore_index=True)
        # save the result df to a CSV file at interval to make sure we don't lose results
        all_result_df.to_csv(all_result_df_path, index=False)

    # convert the results to a DataFrame
    return all_result_df.copy()