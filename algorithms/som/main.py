import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Union, Tuple, Any, Dict
import warnings

warnings.filterwarnings('ignore')

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class SOMFeatureMappingPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using Self-Organizing Maps (SOM).

    This implementation uses SOM to create a 2D topological representation of high-dimensional
    data, then maps the data points to grid positions to create image representations.

    Parameters:
    -----------
    som_shape : tuple of int, default=(20, 20)
        Shape of the SOM grid (height, width). This determines the resolution of the feature map.

    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width). Images will be resized to this size.

    learning_rate : float, default=0.5
        Initial learning rate for SOM training. Decreases over time.

    sigma : float, default=1.0
        Initial radius of influence for SOM neighborhood function.

    num_iterations : int, default=1000
        Number of training iterations for the SOM.

    scaler : str or object, default='standard'
        The scaling method for features. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        - Custom sklearn-compatible scaler object

    neighborhood_function : str, default='gaussian'
        Neighborhood function for SOM. Options: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

    topology : str, default='rectangular'
        Topology of the SOM grid. Options: 'rectangular', 'hexagonal'

    activation_distance : str, default='euclidean'
        Distance metric for SOM activation. Options: 'euclidean', 'cosine', 'manhattan'

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'scalar'

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            som_shape: Tuple[int, int] = (20, 20),
            output_size: Tuple[int, int] = (224, 224),
            learning_rate: float = 0.5,
            sigma: float = 1.0,
            num_iterations: int = 1000,
            scaler: Union[str, Any] = 'standard',
            neighborhood_function: str = 'gaussian',
            topology: str = 'rectangular',
            activation_distance: str = 'euclidean',
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

        # SOM-specific parameters
        self.som_shape = som_shape
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_iterations = num_iterations
        self.scaler = scaler
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance

        # Initialize components
        self._scaler_obj = None
        self._som = None
        self._feature_positions = None
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit the SOM pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector (not used in transformation but kept for sklearn compatibility)

        Returns:
        --------
        self : SOMFeatureMappingPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting SOM Feature Mapping pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"SOM grid shape: {self.som_shape}")
            print(f"Output size: {self.output_size}")
            print(f"Training iterations: {self.num_iterations}")

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

        # Initialize and train SOM
        self._som = MiniSom(
            x=self.som_shape[0],
            y=self.som_shape[1],
            input_len=X_scaled.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_state
        )

        if self.verbose:
            print("Initializing SOM weights...")
        self._som.random_weights_init(X_scaled)

        if self.verbose:
            print("Training SOM...")
        self._som.train(X_scaled, self.num_iterations, verbose=self.verbose)

        # Create feature position mapping
        self._create_feature_positions(X_scaled)

        self._is_fitted = True
        if self.verbose:
            print("SOM pipeline fitted successfully!")

        return self

    def _create_feature_positions(self, X_scaled: np.ndarray):
        """Create a mapping of features to SOM grid positions."""
        # Get the best matching units (BMUs) for each sample
        bmus = []
        for x in X_scaled:
            bmu = self._som.winner(x)
            bmus.append(bmu)

        self._feature_positions = np.array(bmus)

        if self.verbose:
            print(f"Created feature position mapping with {len(self._feature_positions)} samples")

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using SOM feature mapping.

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
            print(f"Transforming data to SOM images...")
            print(f"Input shape: {X.shape}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Transform each sample to an image
        images = []
        for i, x in enumerate(X_scaled):
            img = self._create_som_image(x)
            images.append(img)

        X_img = np.array(images)

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def _create_som_image(self, x: np.ndarray) -> np.ndarray:
        """Create an image representation from a single data sample using SOM."""
        # Get the best matching unit
        bmu = self._som.winner(x)

        # Create base image using SOM distance map
        distance_map = self._som.distance_map()

        # Create activation map - how much each neuron is activated by this sample
        activation_map = np.zeros(self.som_shape)

        # Calculate distances from input to all neurons
        for i in range(self.som_shape[0]):
            for j in range(self.som_shape[1]):
                neuron_weights = self._som.get_weights()[i, j]
                distance = np.linalg.norm(x - neuron_weights)
                # Convert distance to activation (higher activation = lower distance)
                activation_map[i, j] = np.exp(-distance)

        # Normalize activation map
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)

        # Combine distance map and activation map
        combined_map = 0.5 * distance_map + 0.5 * (1 - activation_map)

        # Highlight the BMU
        combined_map[bmu] = combined_map.max()

        # Convert to desired output format
        if self.img_format == 'rgb':
            # Create RGB image using colormap
            img = self._create_rgb_image(combined_map)
        else:
            # Scalar format
            img = combined_map

        # Resize to output size
        img_resized = self._resize_image(img, self.output_size)

        return img_resized

    def _create_rgb_image(self, data_map: np.ndarray) -> np.ndarray:
        """Convert a 2D map to RGB image using colormap."""
        # Normalize to [0, 1]
        normalized = (data_map - data_map.min()) / (data_map.max() - data_map.min() + 1e-8)

        # Apply colormap
        colormap = cm.get_cmap('viridis')
        rgb_image = colormap(normalized)[:, :, :3]  # Remove alpha channel

        return rgb_image

    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size using interpolation."""
        from scipy.ndimage import zoom

        if img.ndim == 2:
            # Grayscale image
            zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
            resized = zoom(img, zoom_factors, order=1)
        else:
            # RGB image
            zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1], 1)
            resized = zoom(img, zoom_factors, order=1)

        return resized

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'som_shape': self.som_shape,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'sigma': self.sigma,
            'num_iterations': self.num_iterations,
            'scaler': self.scaler,
            'neighborhood_function': self.neighborhood_function,
            'topology': self.topology,
            'activation_distance': self.activation_distance,
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

    def visualize_som(self, save_path: Optional[str] = None):
        """
        Visualize the trained SOM grid.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if not self._is_fitted:
            raise ValueError("SOM must be fitted before visualization")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Distance map
        distance_map = self._som.distance_map()
        axes[0].imshow(distance_map, cmap='viridis')
        axes[0].set_title('SOM Distance Map')
        axes[0].axis('off')

        # Activation frequency map
        activation_map = np.zeros(self.som_shape)
        if self._feature_positions is not None:
            for pos in self._feature_positions:
                activation_map[pos[0], pos[1]] += 1

        axes[1].imshow(activation_map, cmap='hot')
        axes[1].set_title('SOM Activation Frequency')
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"SOM visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_som_weights(self) -> np.ndarray:
        """Get the trained SOM weight matrix."""
        if not self._is_fitted:
            raise ValueError("SOM must be fitted before accessing weights")
        return self._som.get_weights()

    def get_feature_positions(self) -> np.ndarray:
        """Get the feature positions mapping."""
        if not self._is_fitted:
            raise ValueError("SOM must be fitted before accessing feature positions")
        return self._feature_positions