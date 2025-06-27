from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def tabular_testing(file_name, target_column, model_configs,n_runs=5, test_size=0.2):
    """
    This function loads a dataset, runs multiple evaluation trials,
    and tracks comprehensive metrics including training time.

    Returns:
        pd.DataFrame: DataFrame with columns ['model_name', 'run', 'accuracy', 'precision', 'recall', 'f1_score', 'train_time']
    """
    # Load the dataset
    df = pd.read_csv(file_name, index_col=0)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if binary or multiclass
    is_binary = len(np.unique(y)) == 2

    # Store all individual results
    results_list = []

    # Run multiple trials
    for run in range(n_runs):
        # Split the data with different random state for each run
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=run, stratify=y
        )

        # Test each model
        for model_name, model_info in model_configs.items():
            model_class = model_info["class"]
            model_params = model_info["params"].copy()

            # Update random state for this run (if model has random_state)
            if "random_state" in model_params:
                model_params["random_state"] = run

            # Instantiate the model
            model = model_class(**model_params)

            # Time the training
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            if is_binary:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

            # Store results as a row
            results_list.append({
                'model_name': model_name,
                'run': run + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df