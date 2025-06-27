import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Optional, Union, Tuple, Any, Dict
import warnings

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class SuperTMLPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using SuperTML methodology.

    SuperTML transforms tabular data into 2D image representations by:
    1. Handling categorical and missing values automatically
    2. Computing feature importance scores
    3. Positioning features on a 2D grid based on importance
    4. Creating image representations where pixel intensity represents feature values

    Parameters:
    -----------
    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)

    scaler : str or object, default='standard'
        The scaling method for features. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        - Custom sklearn-compatible scaler object

    feature_selection : str or int, default='all'
        Feature selection strategy:
        - 'all': Use all features
        - int: Select top k features based on importance
        - 'auto': Automatically determine number of features based on image size

    importance_method : str, default='f_classif'
        Method to compute feature importance:
        - 'f_classif': F-classification score
        - 'mutual_info': Mutual information
        - 'variance': Feature variance

    fill_strategy : str, default='mean'
        Strategy for filling missing values:
        - 'mean': Fill with mean value
        - 'median': Fill with median value
        - 'mode': Fill with most frequent value
        - 'zero': Fill with zero

    categorical_encoding : str, default='label'
        Strategy for encoding categorical variables:
        - 'label': Label encoding
        - 'onehot': One-hot encoding (not recommended for high cardinality)
        - 'target': Target encoding (requires y during fit)

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'grayscale'

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (224, 224),
            scaler: Union[str, Any] = 'standard',
            feature_selection: Union[str, int] = 'auto',
            importance_method: str = 'f_classif',
            fill_strategy: str = 'mean',
            categorical_encoding: str = 'label',
            img_format: str = 'rgb',
            random_state: int = 42,
            verbose: bool = True
    ):
        # Initialize parent class
        super().__init__(
            output_size=output_size,
            img_format=img_format,
            random_state=random_state,
            verbose=verbose
        )

        # SuperTML-specific parameters
        self.scaler = scaler
        self.feature_selection = feature_selection
        self.importance_method = importance_method
        self.fill_strategy = fill_strategy
        self.categorical_encoding = categorical_encoding

        # Internal components
        self._scaler_obj = None
        self._feature_selector = None
        self._label_encoders = {}
        self._feature_positions = None
        self._feature_importance = None
        self._categorical_columns = []
        self._numerical_columns = []
        self._selected_features = None
        self._fill_values = {}
        self._original_feature_names = None

        # Setup components
        self._setup_scaler()

    def _setup_scaler(self):
        """Setup the scaler object."""
        if isinstance(self.scaler, str):
            if self.scaler.lower() == 'standard':
                self._scaler_obj = StandardScaler()
            elif self.scaler.lower() == 'minmax':
                self._scaler_obj = MinMaxScaler()
            elif self.scaler.lower() == 'none':
                self._scaler_obj = None
            else:
                raise ValueError(f"Unknown scaler: {self.scaler}")
        else:
            self._scaler_obj = self.scaler

    def _preprocess_features(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Preprocess features by handling missing values and encoding categoricals."""
        X_processed = X.copy()

        if is_training:
            # Identify column types during training
            self._categorical_columns = []
            self._numerical_columns = []

            for i in range(X.shape[1]):
                col_data = X[:, i]
                # Check if column contains string/object data
                if any(isinstance(val, str) for val in col_data if val is not None and not pd.isna(val)):
                    self._categorical_columns.append(i)
                else:
                    self._numerical_columns.append(i)

        # Process categorical columns
        for col_idx in self._categorical_columns:
            col_data = X_processed[:, col_idx].copy()

            if is_training:
                encoder = LabelEncoder()
                # Convert to string and handle missing values
                col_data_str = []
                for val in col_data:
                    if pd.isna(val) or val is None or val == '':
                        col_data_str.append('__MISSING__')
                    else:
                        col_data_str.append(str(val))

                encoded_values = encoder.fit_transform(col_data_str)
                X_processed[:, col_idx] = encoded_values
                self._label_encoders[col_idx] = encoder
            else:
                if col_idx in self._label_encoders:
                    encoder = self._label_encoders[col_idx]
                    # Convert to string and handle missing values
                    col_data_str = []
                    for val in col_data:
                        if pd.isna(val) or val is None or val == '':
                            col_data_str.append('__MISSING__')
                        else:
                            col_data_str.append(str(val))

                    # Handle unseen categories
                    try:
                        encoded_values = encoder.transform(col_data_str)
                        X_processed[:, col_idx] = encoded_values
                    except ValueError:
                        # Map unseen categories to the most frequent class
                        known_classes = set(encoder.classes_)
                        col_data_mapped = [
                            val if val in known_classes else encoder.classes_[0]
                            for val in col_data_str
                        ]
                        encoded_values = encoder.transform(col_data_mapped)
                        X_processed[:, col_idx] = encoded_values

        # Process numerical columns - handle missing values
        for col_idx in self._numerical_columns:
            col_data = X_processed[:, col_idx].astype(float)
            missing_mask = pd.isna(col_data)

            if missing_mask.any():
                if is_training:
                    # Compute fill value during training
                    valid_data = col_data[~missing_mask]
                    if len(valid_data) == 0:
                        fill_value = 0.0
                    elif self.fill_strategy == 'mean':
                        fill_value = np.mean(valid_data)
                    elif self.fill_strategy == 'median':
                        fill_value = np.median(valid_data)
                    elif self.fill_strategy == 'mode':
                        fill_value = pd.Series(valid_data).mode().iloc[0] if len(valid_data) > 0 else 0.0
                    else:  # zero
                        fill_value = 0.0

                    self._fill_values[col_idx] = fill_value
                else:
                    # Use stored fill value during transform
                    fill_value = self._fill_values.get(col_idx, 0.0)

                col_data[missing_mask] = fill_value
                X_processed[:, col_idx] = col_data

        # Ensure all data is numeric
        return X_processed.astype(float)

    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature importance scores."""
        if self.importance_method == 'f_classif':
            scores, _ = f_classif(X, y)
            scores = np.nan_to_num(scores, nan=0.0)
        elif self.importance_method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=self.random_state)
        elif self.importance_method == 'variance':
            scores = np.var(X, axis=0)
        else:
            raise ValueError(f"Unknown importance method: {self.importance_method}")

        # Normalize scores to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def _select_features(self, X: np.ndarray, y: Optional[np.ndarray] = None, is_fit: bool = True) -> np.ndarray:
        """Select features based on importance."""
        if self.feature_selection == 'all':
            if is_fit:
                self._selected_features = np.arange(X.shape[1])
            return X

        if is_fit:
            # Determine number of features to select
            if self.feature_selection == 'auto':
                # Select features that can fill the image reasonably
                max_features = self.output_size[0] * self.output_size[1]
                n_features = min(X.shape[1], max_features // 4)  # Use 1/4 of image pixels
            elif isinstance(self.feature_selection, int):
                n_features = min(self.feature_selection, X.shape[1])
            else:
                n_features = X.shape[1]

            if y is not None and n_features < X.shape[1]:
                # Use feature selection based on importance
                if self.importance_method in ['f_classif', 'mutual_info']:
                    selector = SelectKBest(
                        score_func=f_classif if self.importance_method == 'f_classif' else mutual_info_classif,
                        k=n_features
                    )
                    X_selected = selector.fit_transform(X, y)
                    self._selected_features = selector.get_support(indices=True)
                else:
                    # Use variance-based selection
                    importance_scores = self._compute_feature_importance(X, y)
                    self._selected_features = np.argsort(importance_scores)[-n_features:]
                    X_selected = X[:, self._selected_features]
            else:
                self._selected_features = np.arange(min(n_features, X.shape[1]))
                X_selected = X[:, self._selected_features]
        else:
            X_selected = X[:, self._selected_features]

        return X_selected

    def _create_feature_positions(self, n_features: int, importance_scores: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """Create 2D positions for features based on importance."""
        positions = {}

        # Sort features by importance (descending)
        sorted_indices = np.argsort(importance_scores)[::-1]

        # Create a spiral pattern starting from center, with most important features closer to center
        height, width = self.output_size
        center_y, center_x = height // 2, width // 2

        # Generate spiral positions
        spiral_positions = []
        y, x = center_y, center_x
        spiral_positions.append((y, x))

        # Spiral outward
        direction = 0  # 0: right, 1: down, 2: left, 3: up
        steps = 1

        while len(spiral_positions) < n_features and len(spiral_positions) < height * width:
            for _ in range(2):  # Two sides of the spiral have the same number of steps
                for _ in range(steps):
                    if direction == 0:  # right
                        x += 1
                    elif direction == 1:  # down
                        y += 1
                    elif direction == 2:  # left
                        x -= 1
                    else:  # up
                        y -= 1

                    if 0 <= y < height and 0 <= x < width:
                        spiral_positions.append((y, x))

                    if len(spiral_positions) >= n_features:
                        break

                direction = (direction + 1) % 4
                if len(spiral_positions) >= n_features:
                    break

            steps += 1

        # Assign positions to features
        for i, feature_idx in enumerate(sorted_indices):
            if i < len(spiral_positions):
                positions[feature_idx] = spiral_positions[i]

        return positions

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector

        Returns:
        --------
        self : SuperTMLPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting SuperTML pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Output size: {self.output_size}")

        # Convert input to numpy array and store feature names
        if isinstance(X, pd.DataFrame):
            self._original_feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X.copy()
            self._original_feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

        # Convert target to numpy array
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values

        # Preprocess features (handle categorical, missing values)
        X_processed = self._preprocess_features(X_array, is_training=True)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.fit_transform(X_processed)
            if self.verbose:
                print(f"Data scaled using {type(self._scaler_obj).__name__}")
        else:
            X_scaled = X_processed
            if self.verbose:
                print("No scaling applied")

        # Feature selection
        X_selected = self._select_features(X_scaled, y, is_fit=True)

        # Compute feature importance
        if y is not None:
            self._feature_importance = self._compute_feature_importance(X_selected, y)
        else:
            # Use variance as importance if no target provided
            self._feature_importance = np.var(X_selected, axis=0)
            if self._feature_importance.max() > 0:
                self._feature_importance = self._feature_importance / self._feature_importance.max()

        # Create feature positions
        n_selected_features = X_selected.shape[1]
        self._feature_positions = self._create_feature_positions(n_selected_features, self._feature_importance)

        self._is_fitted = True

        if self.verbose:
            print(f"Selected {n_selected_features} features")
            print(f"Feature positions created for {len(self._feature_positions)} features")
            print("Pipeline fitted successfully!")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix to transform

        Returns:
        --------
        X_img : ndarray
            Transformed images
        """
        self._validate_fitted()

        if self.verbose:
            print(f"Transforming data to images...")
            print(f"Input shape: {X.shape}")

        # Convert input to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()

        # Preprocess features
        X_processed = self._preprocess_features(X_array, is_training=False)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_processed)
        else:
            X_scaled = X_processed

        # Feature selection
        X_selected = self._select_features(X_scaled, is_fit=False)

        # Create images
        n_samples = X_selected.shape[0]
        height, width = self.output_size

        if self.img_format == 'rgb':
            X_img = np.zeros((n_samples, height, width, 3), dtype=np.float32)
        else:
            X_img = np.zeros((n_samples, height, width), dtype=np.float32)

        # Fill images with feature values
        for sample_idx in range(n_samples):
            for feature_idx, (y, x) in self._feature_positions.items():
                if feature_idx < X_selected.shape[1]:
                    feature_value = X_selected[sample_idx, feature_idx]
                    importance = self._feature_importance[feature_idx]

                    # Normalize feature value to [0, 1] range
                    # Handle different scaling scenarios
                    if self._scaler_obj is not None:
                        # For standardized data, map from roughly [-3, 3] to [0, 1]
                        pixel_intensity = np.clip((feature_value + 3) / 6, 0, 1)
                    else:
                        # For non-scaled data, use min-max normalization
                        col_min = np.min(X_selected[:, feature_idx])
                        col_max = np.max(X_selected[:, feature_idx])
                        if col_max > col_min:
                            pixel_intensity = (feature_value - col_min) / (col_max - col_min)
                        else:
                            pixel_intensity = 0.5  # Constant value case

                    # Apply importance weighting
                    weighted_intensity = pixel_intensity * importance

                    if self.img_format == 'rgb':
                        # Create different color channels based on feature characteristics
                        X_img[sample_idx, y, x, 0] = weighted_intensity  # Red channel
                        X_img[sample_idx, y, x, 1] = weighted_intensity * 0.7  # Green channel
                        X_img[sample_idx, y, x, 2] = weighted_intensity * 0.5  # Blue channel
                    else:
                        X_img[sample_idx, y, x] = weighted_intensity

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'scaler': self.scaler,
            'feature_selection': self.feature_selection,
            'importance_method': self.importance_method,
            'fill_strategy': self.fill_strategy,
            'categorical_encoding': self.categorical_encoding,
            'img_format': self.img_format,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set pipeline parameters."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")

        # Re-setup components if needed
        if 'scaler' in params:
            self._setup_scaler()

        # Update parent class attributes
        if 'output_size' in params:
            super().__setattr__('output_size', params['output_size'])
        if 'img_format' in params:
            super().__setattr__('img_format', params['img_format'])
        if 'random_state' in params:
            super().__setattr__('random_state', params['random_state'])
        if 'verbose' in params:
            super().__setattr__('verbose', params['verbose'])

        # Reset fitted state
        self._is_fitted = False
        return self
