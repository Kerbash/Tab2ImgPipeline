import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import itertools
from typing import Optional, Union, Tuple, Any, Dict
import warnings

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class IGTDPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using IGTD (Image Generator for Tabular Data) methodology.

    IGTD transforms tabular data into images by assigning each feature to a unique pixel position such that
    similar features are assigned to neighboring pixels in the image representation.

    Parameters:
    -----------
    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)

    distance_metric : str, default='euclidean'
        Distance metric for calculating feature similarities.
        Options: 'euclidean', 'manhattan', 'cosine', 'correlation'

    pixel_distance_metric : str, default='euclidean'
        Distance metric for calculating pixel distances in the image.
        Options: 'euclidean', 'manhattan'

    scaler : str or object, default='standard'
        The scaling method for features. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        - Custom sklearn-compatible scaler object

    optimization_method : str, default='greedy'
        Method for optimizing feature-to-pixel assignment.
        Options: 'greedy', 'hungarian' (for smaller feature sets)

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'scalar', 'grayscale'

    normalize_image : bool, default=True
        Whether to normalize pixel values to [0, 1] range

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (224, 224),
            distance_metric: str = 'euclidean',
            pixel_distance_metric: str = 'euclidean',
            scaler: Union[str, Any] = 'standard',
            optimization_method: str = 'greedy',
            img_format: str = 'rgb',
            normalize_image: bool = True,
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

        # IGTD-specific parameters
        self.distance_metric = distance_metric
        self.pixel_distance_metric = pixel_distance_metric
        self.scaler = scaler
        self.optimization_method = optimization_method
        self.normalize_image = normalize_image

        # Initialize components
        self._scaler_obj = None
        self._feature_to_pixel_mapping = None
        self._pixel_coordinates = None
        self._n_features = None

        # Setup scaler
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

    def _generate_pixel_coordinates(self) -> np.ndarray:
        """Generate all possible pixel coordinates for the image."""
        height, width = self.output_size
        coordinates = []

        for i in range(height):
            for j in range(width):
                coordinates.append([i, j])

        return np.array(coordinates)

    def _calculate_feature_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between features."""
        # Transpose to get features as rows
        X_features = X.T

        # Calculate pairwise distances between features
        distances = pairwise_distances(X_features, metric=self.distance_metric)

        return distances

    def _calculate_pixel_distances(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between pixel coordinates."""
        if self.pixel_distance_metric == 'euclidean':
            distances = pairwise_distances(pixel_coords, metric='euclidean')
        elif self.pixel_distance_metric == 'manhattan':
            distances = pairwise_distances(pixel_coords, metric='manhattan')
        else:
            raise ValueError(f"Unknown pixel distance metric: {self.pixel_distance_metric}")

        return distances

    def _optimize_assignment_greedy(self, feature_distances: np.ndarray,
                                    pixel_distances: np.ndarray) -> np.ndarray:
        """
        Optimize feature-to-pixel assignment using greedy approach.

        This is a simplified version of the IGTD optimization that works well
        for most cases and is computationally efficient.
        """
        n_features = feature_distances.shape[0]
        n_pixels = pixel_distances.shape[0]

        if n_features > n_pixels:
            raise ValueError(f"Number of features ({n_features}) exceeds number of pixels ({n_pixels})")

        # Initialize assignment
        assignment = np.zeros(n_features, dtype=int)
        used_pixels = set()

        # Start with the first feature, assign to center pixel
        center_pixel = n_pixels // 2
        assignment[0] = center_pixel
        used_pixels.add(center_pixel)

        # For each remaining feature, find the best pixel
        for feature_idx in range(1, n_features):
            best_pixel = None
            best_score = float('inf')

            # Consider all unused pixels
            for pixel_idx in range(n_pixels):
                if pixel_idx in used_pixels:
                    continue

                # Calculate score for this pixel assignment
                score = 0
                for prev_feature_idx in range(feature_idx):
                    prev_pixel_idx = assignment[prev_feature_idx]

                    # Add difference between feature distance and pixel distance
                    feature_dist = feature_distances[feature_idx, prev_feature_idx]
                    pixel_dist = pixel_distances[pixel_idx, prev_pixel_idx]
                    score += abs(feature_dist - pixel_dist)

                if score < best_score:
                    best_score = score
                    best_pixel = pixel_idx

            if best_pixel is not None:
                assignment[feature_idx] = best_pixel
                used_pixels.add(best_pixel)
            else:
                # Fallback: assign to first available pixel
                for pixel_idx in range(n_pixels):
                    if pixel_idx not in used_pixels:
                        assignment[feature_idx] = pixel_idx
                        used_pixels.add(pixel_idx)
                        break

        return assignment

    def _optimize_assignment_hungarian(self, feature_distances: np.ndarray,
                                       pixel_distances: np.ndarray) -> np.ndarray:
        """
        Optimize feature-to-pixel assignment using Hungarian algorithm.

        This is more accurate but computationally expensive for large feature sets.
        """
        n_features = feature_distances.shape[0]
        n_pixels = pixel_distances.shape[0]

        if n_features > n_pixels:
            raise ValueError(f"Number of features ({n_features}) exceeds number of pixels ({n_pixels})")

        # Create cost matrix
        cost_matrix = np.zeros((n_features, n_pixels))

        for i in range(n_features):
            for j in range(n_pixels):
                # Calculate cost as sum of distance differences
                cost = 0
                for k in range(n_features):
                    if k != i:
                        for l in range(n_pixels):
                            if l != j:
                                feature_dist = feature_distances[i, k]
                                pixel_dist = pixel_distances[j, l]
                                cost += abs(feature_dist - pixel_dist)

                cost_matrix[i, j] = cost

        # Solve assignment problem
        feature_indices, pixel_indices = linear_sum_assignment(cost_matrix)

        # Create assignment array
        assignment = np.zeros(n_features, dtype=int)
        assignment[feature_indices] = pixel_indices

        return assignment

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the IGTD pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector (not used in transformation but kept for sklearn compatibility)

        Returns:
        --------
        self : IGTDPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting IGTD pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Distance metric: {self.distance_metric}")
            print(f"Output size: {self.output_size}")
            print(f"Optimization method: {self.optimization_method}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)
        self._n_features = X_array.shape[1]

        # Check if we have too many features for the image size
        max_pixels = self.output_size[0] * self.output_size[1]
        if self._n_features > max_pixels:
            raise ValueError(f"Number of features ({self._n_features}) exceeds maximum pixels ({max_pixels}). "
                             f"Consider reducing features or increasing output_size.")

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.fit_transform(X_array)
            if self.verbose:
                print(f"Data scaled using {type(self._scaler_obj).__name__}")
        else:
            X_scaled = X_array
            if self.verbose:
                print("No scaling applied")

        # Generate pixel coordinates
        self._pixel_coordinates = self._generate_pixel_coordinates()

        # Calculate feature distances
        if self.verbose:
            print("Calculating feature distances...")
        feature_distances = self._calculate_feature_distances(X_scaled)

        # Calculate pixel distances
        if self.verbose:
            print("Calculating pixel distances...")
        pixel_distances = self._calculate_pixel_distances(self._pixel_coordinates)

        # Optimize feature-to-pixel assignment
        if self.verbose:
            print(f"Optimizing feature-to-pixel assignment using {self.optimization_method} method...")

        if self.optimization_method == 'greedy':
            self._feature_to_pixel_mapping = self._optimize_assignment_greedy(
                feature_distances, pixel_distances)
        elif self.optimization_method == 'hungarian':
            if self._n_features > 100:
                warnings.warn("Hungarian algorithm may be slow for large feature sets. Consider using 'greedy' method.")
            self._feature_to_pixel_mapping = self._optimize_assignment_hungarian(
                feature_distances, pixel_distances)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        self._is_fitted = True
        if self.verbose:
            print("IGTD pipeline fitted successfully!")
            print(f"Feature-to-pixel mapping created for {self._n_features} features")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using IGTD.

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
            print(f"Transforming data to images using IGTD...")
            print(f"Input shape: {X.shape}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        if X_array.shape[1] != self._n_features:
            raise ValueError(f"Number of features ({X_array.shape[1]}) does not match "
                             f"fitted features ({self._n_features})")

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Transform to images
        n_samples = X_scaled.shape[0]
        height, width = self.output_size

        if self.img_format == 'rgb':
            images = np.zeros((n_samples, height, width, 3))
        else:
            images = np.zeros((n_samples, height, width))

        for sample_idx in range(n_samples):
            # Create image for this sample
            if self.img_format == 'rgb':
                image = np.zeros((height, width, 3))
            else:
                image = np.zeros((height, width))

            # Assign feature values to pixels according to mapping
            for feature_idx in range(self._n_features):
                pixel_idx = self._feature_to_pixel_mapping[feature_idx]
                pixel_coords = self._pixel_coordinates[pixel_idx]
                row, col = pixel_coords

                feature_value = X_scaled[sample_idx, feature_idx]

                if self.img_format == 'rgb':
                    # For RGB, we can use different strategies
                    # Here we'll use the feature value for all channels
                    image[row, col, :] = feature_value
                else:
                    image[row, col] = feature_value

            # Normalize if requested
            if self.normalize_image:
                if self.img_format == 'rgb':
                    # Normalize each channel
                    for c in range(3):
                        channel = image[:, :, c]
                        if channel.max() > channel.min():
                            image[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
                else:
                    if image.max() > image.min():
                        image = (image - image.min()) / (image.max() - image.min())

            images[sample_idx] = image

        if self.verbose:
            print(f"Output image shape: {images.shape}")

        return images

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'distance_metric': self.distance_metric,
            'pixel_distance_metric': self.pixel_distance_metric,
            'scaler': self.scaler,
            'optimization_method': self.optimization_method,
            'img_format': self.img_format,
            'normalize_image': self.normalize_image,
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

    def get_feature_mapping(self) -> Optional[np.ndarray]:
        """
        Get the feature-to-pixel mapping.

        Returns:
        --------
        mapping : ndarray or None
            Array where mapping[i] gives the pixel index for feature i
        """
        return self._feature_to_pixel_mapping.copy() if self._is_fitted else None

    def visualize_mapping(self) -> Optional[np.ndarray]:
        """
        Create a visualization of the feature-to-pixel mapping.

        Returns:
        --------
        mapping_image : ndarray or None
            Image showing which pixels are assigned to features
        """
        if not self._is_fitted:
            return None

        height, width = self.output_size
        mapping_image = np.zeros((height, width))

        for feature_idx in range(self._n_features):
            pixel_idx = self._feature_to_pixel_mapping[feature_idx]
            pixel_coords = self._pixel_coordinates[pixel_idx]
            row, col = pixel_coords
            mapping_image[row, col] = feature_idx + 1  # +1 so features are numbered from 1

        return mapping_image
