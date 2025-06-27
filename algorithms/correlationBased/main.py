import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from typing import Optional, Union, Tuple, Any, Dict
import warnings
from itertools import combinations

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class CorrelationPixelMappingPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using correlation-based pixel mapping.

    This method positions features in a 2D image space based on their correlation relationships,
    where highly correlated features are placed close together and uncorrelated features
    are placed farther apart.

    Parameters:
    -----------
    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)

    correlation_method : str, default='pearson'
        Method to compute correlations. Options: 'pearson', 'spearman', 'kendall'

    distance_metric : str, default='correlation'
        Distance metric for feature positioning. Options: 'correlation', 'euclidean', 'cosine'

    clustering_method : str, default='ward'
        Hierarchical clustering method. Options: 'ward', 'complete', 'average', 'single'

    positioning_strategy : str, default='spiral'
        Strategy for positioning features in 2D space. Options: 'spiral', 'grid', 'hierarchical'

    normalization : str, default='minmax'
        Pixel intensity normalization method. Options: 'minmax', 'standard', 'none'

    scaler : str or object, default='standard'
        The scaling method for features before correlation computation

    img_format : str, default='rgb'
        Output image format

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (224, 224),
            correlation_method: str = 'pearson',
            distance_metric: str = 'correlation',
            clustering_method: str = 'ward',
            positioning_strategy: str = 'spiral',
            normalization: str = 'minmax',
            scaler: Union[str, Any] = 'standard',
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

        # Correlation-based mapping parameters
        self.correlation_method = correlation_method
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method
        self.positioning_strategy = positioning_strategy
        self.normalization = normalization
        self.scaler = scaler

        # Initialize components
        self._scaler_obj = None
        self._correlation_matrix = None
        self._feature_positions = None
        self._pixel_intensities_scaler = None

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

    def _compute_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix between features."""
        if self.correlation_method == 'pearson':
            corr_matrix = np.corrcoef(X.T)
        elif self.correlation_method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(X, axis=0)
        elif self.correlation_method == 'kendall':
            from scipy.stats import kendalltau
            n_features = X.shape[1]
            corr_matrix = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        tau, _ = kendalltau(X[:, i], X[:, j])
                        corr_matrix[i, j] = tau
        else:
            raise ValueError(f"Unknown correlation method: {self.correlation_method}")

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        return corr_matrix

    def _compute_distance_matrix(self, X: np.ndarray, corr_matrix: np.ndarray) -> np.ndarray:
        """Compute distance matrix from correlation matrix."""
        if self.distance_metric == 'correlation':
            # Convert correlation to distance: distance = 1 - |correlation|
            distance_matrix = 1 - np.abs(corr_matrix)
        elif self.distance_metric == 'euclidean':
            distance_matrix = squareform(pdist(X.T, metric='euclidean'))
        elif self.distance_metric == 'cosine':
            similarity_matrix = cosine_similarity(X.T)
            distance_matrix = 1 - similarity_matrix
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distance_matrix

    def _generate_spiral_positions(self, n_features: int) -> np.ndarray:
        """Generate positions using spiral arrangement."""
        positions = np.zeros((n_features, 2))

        # Create spiral coordinates
        angle_step = 2 * np.pi / max(6, int(np.sqrt(n_features)))
        radius_step = min(self.output_size) / (2 * max(1, int(np.sqrt(n_features))))

        center_x, center_y = self.output_size[1] // 2, self.output_size[0] // 2

        for i in range(n_features):
            angle = i * angle_step
            radius = (i // 6) * radius_step + radius_step

            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Ensure positions are within bounds
            x = max(0, min(self.output_size[1] - 1, int(x)))
            y = max(0, min(self.output_size[0] - 1, int(y)))

            positions[i] = [y, x]

        return positions.astype(int)

    def _generate_grid_positions(self, n_features: int) -> np.ndarray:
        """Generate positions using grid arrangement."""
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_features)))
        step_x = self.output_size[1] // grid_size
        step_y = self.output_size[0] // grid_size

        positions = np.zeros((n_features, 2))

        for i in range(n_features):
            row = i // grid_size
            col = i % grid_size

            x = min(col * step_x + step_x // 2, self.output_size[1] - 1)
            y = min(row * step_y + step_y // 2, self.output_size[0] - 1)

            positions[i] = [y, x]

        return positions.astype(int)

    def _generate_hierarchical_positions(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Generate positions using hierarchical clustering."""
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method=self.clustering_method)

        # Get cluster assignments
        n_clusters = min(int(np.sqrt(len(distance_matrix))), len(distance_matrix))
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        positions = np.zeros((len(distance_matrix), 2))

        # Assign positions based on clusters
        cluster_centers = self._get_cluster_centers(n_clusters)

        for cluster_id in range(1, n_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            center = cluster_centers[cluster_id - 1]

            # Arrange features within cluster
            for i, idx in enumerate(cluster_indices):
                offset_angle = 2 * np.pi * i / len(cluster_indices)
                offset_radius = min(20, max(5, len(cluster_indices)))

                x = center[1] + offset_radius * np.cos(offset_angle)
                y = center[0] + offset_radius * np.sin(offset_angle)

                # Ensure positions are within bounds
                x = max(0, min(self.output_size[1] - 1, int(x)))
                y = max(0, min(self.output_size[0] - 1, int(y)))

                positions[idx] = [y, x]

        return positions.astype(int)

    def _get_cluster_centers(self, n_clusters: int) -> np.ndarray:
        """Get cluster center positions."""
        centers = []
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        step_x = self.output_size[1] // (grid_size + 1)
        step_y = self.output_size[0] // (grid_size + 1)

        for i in range(n_clusters):
            row = i // grid_size
            col = i % grid_size

            x = (col + 1) * step_x
            y = (row + 1) * step_y

            centers.append([y, x])

        return np.array(centers)

    def _position_features(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Position features in 2D space based on correlation relationships."""
        n_features = len(distance_matrix)

        if self.positioning_strategy == 'spiral':
            positions = self._generate_spiral_positions(n_features)
        elif self.positioning_strategy == 'grid':
            positions = self._generate_grid_positions(n_features)
        elif self.positioning_strategy == 'hierarchical':
            positions = self._generate_hierarchical_positions(distance_matrix)
        else:
            raise ValueError(f"Unknown positioning strategy: {self.positioning_strategy}")

        return positions

    def _setup_pixel_intensity_normalization(self, X: np.ndarray):
        """Setup normalization for pixel intensities."""
        if self.normalization == 'minmax':
            self._pixel_intensities_scaler = MinMaxScaler(feature_range=(0, 255))
            self._pixel_intensities_scaler.fit(X)
        elif self.normalization == 'standard':
            self._pixel_intensities_scaler = StandardScaler()
            self._pixel_intensities_scaler.fit(X)
        else:
            self._pixel_intensities_scaler = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector (not used in transformation but kept for sklearn compatibility)

        Returns:
        --------
        self : CorrelationPixelMappingPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting Correlation-Based Pixel Mapping pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Correlation method: {self.correlation_method}")
            print(f"Distance metric: {self.distance_metric}")
            print(f"Positioning strategy: {self.positioning_strategy}")
            print(f"Output size: {self.output_size}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Check if we have enough pixels for all features
        total_pixels = self.output_size[0] * self.output_size[1]
        if X_array.shape[1] > total_pixels:
            warnings.warn(f"Number of features ({X_array.shape[1]}) exceeds available pixels ({total_pixels}). "
                          f"Some features may overlap in the image representation.")

        # Scale the data for correlation computation
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.fit_transform(X_array)
            if self.verbose:
                print(f"Data scaled using {type(self._scaler_obj).__name__}")
        else:
            X_scaled = X_array
            if self.verbose:
                print("No scaling applied for correlation computation")

        # Compute correlation matrix
        if self.verbose:
            print("Computing correlation matrix...")
        self._correlation_matrix = self._compute_correlation_matrix(X_scaled)

        # Compute distance matrix
        if self.verbose:
            print("Computing distance matrix...")
        distance_matrix = self._compute_distance_matrix(X_scaled, self._correlation_matrix)

        # Position features in 2D space
        if self.verbose:
            print("Positioning features in 2D space...")
        self._feature_positions = self._position_features(distance_matrix)

        # Setup pixel intensity normalization
        self._setup_pixel_intensity_normalization(X_array)

        self._is_fitted = True
        if self.verbose:
            print("Pipeline fitted successfully!")
            print(f"Feature positions shape: {self._feature_positions.shape}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using correlation-based pixel mapping.

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

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Apply pixel intensity normalization
        if self._pixel_intensities_scaler is not None:
            X_normalized = self._pixel_intensities_scaler.transform(X_array)
        else:
            X_normalized = X_array.copy()

        # Create images
        n_samples = X_array.shape[0]
        n_features = X_array.shape[1]

        if self.img_format == 'rgb':
            images = np.zeros((n_samples, self.output_size[0], self.output_size[1], 3))
        else:
            images = np.zeros((n_samples, self.output_size[0], self.output_size[1]))

        # Map features to pixels
        for sample_idx in range(n_samples):
            sample_data = X_normalized[sample_idx]

            for feature_idx in range(min(n_features, len(self._feature_positions))):
                pos_y, pos_x = self._feature_positions[feature_idx]
                pixel_value = sample_data[feature_idx]

                # Ensure pixel value is in valid range
                if self.normalization == 'none':
                    pixel_value = np.clip(pixel_value, 0, 255)

                if self.img_format == 'rgb':
                    # For RGB, replicate value across all channels
                    images[sample_idx, pos_y, pos_x, :] = pixel_value
                else:
                    images[sample_idx, pos_y, pos_x] = pixel_value

        # Convert to appropriate data type
        if self.img_format == 'rgb':
            images = images.astype(np.uint8)
        else:
            images = images.astype(np.float32)

        if self.verbose:
            print(f"Output image shape: {images.shape}")

        return images

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'correlation_method': self.correlation_method,
            'distance_metric': self.distance_metric,
            'clustering_method': self.clustering_method,
            'positioning_strategy': self.positioning_strategy,
            'normalization': self.normalization,
            'scaler': self.scaler,
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

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get the computed correlation matrix."""
        return self._correlation_matrix

    def get_feature_positions(self) -> Optional[np.ndarray]:
        """Get the feature positions in the 2D image space."""
        return self._feature_positions

    def visualize_feature_mapping(self) -> None:
        """Visualize how features are mapped to pixel positions."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before visualization.")

        try:
            import matplotlib.pyplot as plt

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot correlation matrix
            im1 = ax1.imshow(self._correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_title('Feature Correlation Matrix')
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Features')
            plt.colorbar(im1, ax=ax1)

            # Plot feature positions
            positions_plot = np.zeros(self.output_size)
            for i, (y, x) in enumerate(self._feature_positions):
                positions_plot[y, x] = i + 1

            im2 = ax2.imshow(positions_plot, cmap='viridis')
            ax2.set_title('Feature Positions in Image Space')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            plt.colorbar(im2, ax=ax2, label='Feature Index')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available. Cannot create visualization.")
            print(f"Feature positions shape: {self._feature_positions.shape}")
            print(f"Correlation matrix shape: {self._correlation_matrix.shape}")