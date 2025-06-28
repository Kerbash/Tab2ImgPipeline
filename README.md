# Tabular-to-Image Classification Pipeline

A machine learning pipeline that transforms tabular data into images using various algorithms and performs classification using both traditional ML models and AutoKeras CNNs.

## Installation

```bash
pip install -r requirements.txt
```

## Project Overview

This project provides a framework for:

1. Transforming tabular data into image representations using multiple algorithms
2. Running traditional ML models (Random Forest, SVM) on the original tabular data
3. Training CNN models on the generated images using AutoKeras
4. Comparing performance metrics across all approaches

## Project Structure

```
├── algorithms/           # Image transformation algorithms
│   ├── correlationBased/ # Correlation-based pixel mapping
│   ├── deepinsight/      # DeepInsight algorithm
│   ├── gramMatrix/       # Gram Matrix transformation
│   ├── igtd/             # IGTD algorithm
│   ├── random/           # Random stacking
│   ├── refined/          # REFINED algorithm
│   ├── som/              # SOM feature mapping
│   └── supertml/         # SuperTML algorithm
├── datasets/             # Dataset storage
│   └── sample/           # Example dataset
│       ├── algorithm1/   # Generated images for algorithm1
│       ├── algorithm2/   # Generated images for algorithm2
│       └── ...
├── libs/                 # Utility functions
├── results/              # Experiment results
├── preprocess_main.ipynb # Main preprocessing notebook
└── binary_ml_pipeline.ipynb # Main classification pipeline
```

## Usage

### 1. Preprocess tabular data into images

Run the preprocessing notebook to transform tabular data into images:

```bash
jupyter notebook preprocess_main.ipynb
```

Configure the `DATASETS` variable to point to your CSV files.

### 2. Run the classification pipeline

Execute the classification pipeline to train and evaluate models:

```bash
jupyter notebook binary_ml_pipeline.ipynb
```

## Algorithms

The project implements several tabular-to-image transformation techniques:

- **Correlation-based Pixel Mapping**: Maps features to pixels based on correlation
- **DeepInsight**: Uses t-SNE to project features into 2D space
- **Gram Matrix**: Transforms data using Gram matrix representations
- **IGTD**: Information Geometry-based feature mapping
- **Random**: Random feature stacking
- **REFINED**: Feature arrangement by relative importance
- **SOM**: Self-organizing maps for feature mapping
- **SuperTML**: Character-based feature visualization

## Results Output

Results are saved in the `results/` directory with the following structure:

```
results/
└── dataset_name/
    └── timestamp/
        └── results.csv  # Contains metrics for all algorithms
```

The results CSV includes the following metrics for each model:
- Accuracy
- Precision
- Recall
- F1-score
- Training time

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow
- autokeras
- Pillow
- tqdm