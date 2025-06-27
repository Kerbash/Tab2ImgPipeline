import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Union, Tuple, Any, Dict
import warnings

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class REFINEDPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using REFINED methodology.

    REFINED (Representation Enhancement For INterpreted Data) uses Bayesian variant
    of Metric Multidimensional Scaling (BMDS) to transform tabular data into images
    while preserving spatial correlations between features.

    Parameters:
    -----------
    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)

    n_components : int, default=2
        Number of dimensions for MDS embedding (typically 2 for 2D images)

    distance_metric : str, default='euclidean'
        Distance metric for MDS. Options: 'euclidean', 'manhattan', 'cosine', 'correlation'

    mds_metric : bool, default=True
        Whether to use metric MDS (True) or non-metric MDS (False)

    scaler : str or object, default='standard'
        The scaling method for features. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        - Custom sklearn-compatible scaler object

    bayesian_prior : bool, default=True
        Whether to apply Bayesian-inspired prior regularization

    noise_factor : float, default=0.1
        Noise factor for Bayesian regularization (0.0 to 1.0)

    interpolation_method : str, default='gaussian'
        Method for converting scattered points to image grid
        Options: 'gaussian', 'nearest', 'linear'

    sigma : float, default=1.0
        Standard deviation for Gaussian interpolation

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'scalar', 'pytorch'

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (224, 224),
            n_components: int = 2,
            distance_metric: str = 'euclidean',
            mds_metric: bool = True,
            scaler: Union[str, Any] = 'standard',
            bayesian_prior: bool = True,
            noise_factor: float = 0.1,
            interpolation_method: str = 'gaussian',
            sigma: float = 1.0,
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

        # REFINED-specific parameters
        self.n_components = n_components
        self.distance_metric = distance_metric
        self.mds_metric = mds_metric
        self.scaler = scaler
        self.bayesian_prior = bayesian_prior
        self.noise_factor = noise_factor
        self.interpolation_method = interpolation_method
        self.sigma = sigma

        # Initialize components
        self._scaler_obj = None
        self._mds = None
        self._feature_positions = None
        self._distance_matrix = None

        # Setup components
        self._setup_scaler()
        self._setup_mds()

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

    def _setup_mds(self):
        """Setup the MDS object."""
        self._mds = MDS(
            n_components=self.n_components,
            metric=self.mds_metric,
            random_state=self.random_state,
            max_iter=1000,
            eps=1e-6,
            dissimilarity='precomputed'
        )

    def _compute_feature_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between features using specified metric.

        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)

        Returns:
        --------
        distance_matrix : ndarray
            Pairwise distance matrix between features
        """
        # Transpose to get features x samples for feature-wise distances
        X_features = X.T

        if self.distance_metric == 'euclidean':
            distances = pairwise_distances(X_features, metric='euclidean')
        elif self.distance_metric == 'manhattan':
            distances = pairwise_distances(X_features, metric='manhattan')
        elif self.distance_metric == 'cosine':
            distances = pairwise_distances(X_features, metric='cosine')
        elif self.distance_metric == 'correlation':
            # Correlation distance = 1 - correlation coefficient
            corr_matrix = np.corrcoef(X_features)
            distances = 1 - np.abs(corr_matrix)
            # Handle NaN values
            distances = np.nan_to_num(distances, nan=1.0)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _apply_bayesian_regularization(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Apply Bayesian-inspired regularization to the distance matrix.

        Parameters:
        -----------
        distance_matrix : ndarray
            Original distance matrix

        Returns:
        --------
        regularized_matrix : ndarray
            Regularized distance matrix
        """
        if not self.bayesian_prior:
            return distance_matrix

        # Add small amount of noise for regularization (Bayesian prior)
        np.random.seed(self.random_state)
        noise = np.random.normal(0, self.noise_factor, distance_matrix.shape)
        noise = (noise + noise.T) / 2  # Make symmetric
        np.fill_diagonal(noise, 0)  # Keep diagonal as zero

        regularized_matrix = distance_matrix + np.abs(noise)

        # Ensure positive semi-definite property
        eigenvals, eigenvecs = np.linalg.eigh(regularized_matrix)
        eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive eigenvalues
        regularized_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return regularized_matrix

    def _embed_features(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Embed features in 2D space using MDS.

        Parameters:
        -----------
        distance_matrix : ndarray
            Distance matrix between features

        Returns:
        --------
        positions : ndarray
            2D positions of features
        """
        try:
            positions = self._mds.fit_transform(distance_matrix)
        except ValueError as e:
            if self.verbose:
                print(f"MDS failed with metric={self.mds_metric}, trying non-metric MDS")
            # Fallback to non-metric MDS
            self._mds.metric = False
            positions = self._mds.fit_transform(distance_matrix)

        return positions

    def _create_image_grid(self, feature_positions: np.ndarray, feature_values: np.ndarray) -> np.ndarray:
        """
        Create image by interpolating feature values on a grid based on their positions.

        Parameters:
        -----------
        feature_positions : ndarray
            2D positions of features (n_features x 2)
        feature_values : ndarray
            Values of features for current sample (n_features,)

        Returns:
        --------
        image : ndarray
            Generated image
        """
        height, width = self.output_size

        # Create coordinate grids
        x = np.linspace(feature_positions[:, 0].min(), feature_positions[:, 0].max(), width)
        y = np.linspace(feature_positions[:, 1].min(), feature_positions[:, 1].max(), height)
        xx, yy = np.meshgrid(x, y)

        # Initialize image
        image = np.zeros((height, width))

        if self.interpolation_method == 'gaussian':
            # Gaussian interpolation
            for i, (fx, fy) in enumerate(feature_positions):
                # Gaussian kernel centered at feature position
                gaussian = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * self.sigma ** 2))
                image += feature_values[i] * gaussian

        elif self.interpolation_method == 'nearest':
            # Nearest neighbor interpolation
            for i in range(height):
                for j in range(width):
                    grid_point = np.array([xx[i, j], yy[i, j]])
                    distances = np.linalg.norm(feature_positions - grid_point, axis=1)
                    nearest_idx = np.argmin(distances)
                    image[i, j] = feature_values[nearest_idx]

        elif self.interpolation_method == 'linear':
            # Inverse distance weighted interpolation
            for i in range(height):
                for j in range(width):
                    grid_point = np.array([xx[i, j], yy[i, j]])
                    distances = np.linalg.norm(feature_positions - grid_point, axis=1)
                    distances = np.maximum(distances, 1e-10)  # Avoid division by zero
                    weights = 1.0 / distances
                    weights /= np.sum(weights)
                    image[i, j] = np.sum(weights * feature_values)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")

        return image

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the REFINED pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector (not used in transformation but kept for sklearn compatibility)

        Returns:
        --------
        self : REFINEDPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting REFINED pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Distance metric: {self.distance_metric}")
            print(f"MDS metric: {self.mds_metric}")
            print(f"Bayesian prior: {self.bayesian_prior}")
            print(f"Output size: {self.output_size}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.fit_transform(X_array)
            if self.verbose:
                print(f"Data scaled using {type(self._scaler_obj).__name__}")
        else:
            X_scaled = X_array
            if self.verbose:
                print("No scaling applied")

        # Compute feature distances
        if self.verbose:
            print("Computing feature distance matrix...")
        self._distance_matrix = self._compute_feature_distances(X_scaled)

        # Apply Bayesian regularization
        if self.bayesian_prior:
            if self.verbose:
                print("Applying Bayesian regularization...")
            self._distance_matrix = self._apply_bayesian_regularization(self._distance_matrix)

        # Embed features in 2D space using MDS
        if self.verbose:
            print("Embedding features using MDS...")
        self._feature_positions = self._embed_features(self._distance_matrix)

        # Normalize positions to [0, 1] range
        pos_min = self._feature_positions.min(axis=0)
        pos_max = self._feature_positions.max(axis=0)
        self._feature_positions = (self._feature_positions - pos_min) / (pos_max - pos_min + 1e-10)

        self._is_fitted = True
        if self.verbose:
            print("REFINED pipeline fitted successfully!")
            print(f"Feature positions shape: {self._feature_positions.shape}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using REFINED methodology.

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
            print(f"Transforming data to images using REFINED...")
            print(f"Input shape: {X.shape}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Scale the data using fitted scaler
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Generate images for each sample
        n_samples = X_scaled.shape[0]
        height, width = self.output_size

        if self.img_format == 'rgb':
            X_img = np.zeros((n_samples, height, width, 3))
        else:
            X_img = np.zeros((n_samples, height, width))

        for i in range(n_samples):
            if self.verbose and (i + 1) % max(1, n_samples // 10) == 0:
                print(f"Processing sample {i + 1}/{n_samples}")

            # Get feature values for current sample
            feature_values = X_scaled[i]

            # Create image using interpolation
            image = self._create_image_grid(self._feature_positions, feature_values)

            # Normalize image to [0, 1]
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min())
            else:
                image = np.zeros_like(image)

            if self.img_format == 'rgb':
                # Convert to RGB by replicating across channels
                X_img[i] = np.stack([image, image, image], axis=-1)
            else:
                X_img[i] = image

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'n_components': self.n_components,
            'distance_metric': self.distance_metric,
            'mds_metric': self.mds_metric,
            'scaler': self.scaler,
            'bayesian_prior': self.bayesian_prior,
            'noise_factor': self.noise_factor,
            'interpolation_method': self.interpolation_method,
            'sigma': self.sigma,
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
        if any(p in params for p in ['n_components', 'mds_metric', 'random_state']):
            self._setup_mds()

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

    def plot_feature_positions(self, feature_names: Optional[list] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the 2D positions of features after MDS embedding.

        Parameters:
        -----------
        feature_names : list, optional
            Names of features for labeling
        figsize : tuple
            Figure size for the plot
        """
        self._validate_fitted()

        plt.figure(figsize=figsize)
        scatter = plt.scatter(self._feature_positions[:, 0], self._feature_positions[:, 1],
                              c=range(len(self._feature_positions)), cmap='viridis', alpha=0.7)

        if feature_names is not None and len(feature_names) == len(self._feature_positions):
            for i, name in enumerate(feature_names):
                plt.annotate(name, (self._feature_positions[i, 0], self._feature_positions[i, 1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.colorbar(scatter, label='Feature Index')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.title('REFINED Feature Positions in 2D Space')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_feature_positions(self) -> np.ndarray:
        """
        Get the 2D positions of features after MDS embedding.

        Returns:
        --------
        positions : ndarray
            2D positions of features
        """
        self._validate_fitted()
        return self._feature_positions.copy()
