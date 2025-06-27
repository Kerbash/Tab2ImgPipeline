import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Union, Tuple, Any, Dict
import cv2
from PIL import Image

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class GramMatrixPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using Gram Matrix methodology.

    The Gram matrix captures correlations and relationships between features by computing
    inner products, which can then be visualized as an image representation.

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

    gram_method : str, default='correlation'
        Method to compute the Gram matrix. Options:
        - 'correlation': Feature correlation matrix
        - 'covariance': Feature covariance matrix
        - 'inner_product': Direct inner product of features
        - 'rbf_kernel': RBF kernel matrix
        - 'polynomial_kernel': Polynomial kernel matrix

    feature_reduction : str or None, default=None
        Feature reduction method before computing Gram matrix. Options:
        - None: No reduction
        - 'pca': Use PCA to reduce features
        - 'kmeans': Use K-means clustering to group features

    n_components : int, default=None
        Number of components for feature reduction (if applicable)

    kernel_params : dict, default={}
        Parameters for kernel methods (gamma for RBF, degree for polynomial)

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
            scaler: Union[str, Any] = 'standard',
            gram_method: str = 'correlation',
            feature_reduction: Optional[str] = None,
            n_components: Optional[int] = None,
            kernel_params: Dict[str, Any] = {},
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

        # Gram Matrix-specific parameters
        self.scaler = scaler
        self.gram_method = gram_method
        self.feature_reduction = feature_reduction
        self.n_components = n_components
        self.kernel_params = kernel_params

        # Initialize components
        self._scaler_obj = None
        self._reducer = None
        self._gram_matrices = None

        # Setup scaler and reducer
        self._setup_scaler()
        self._setup_reducer()

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

    def _setup_reducer(self):
        """Setup the feature reduction object."""
        if self.feature_reduction is None:
            self._reducer = None
        elif self.feature_reduction.lower() == 'pca':
            n_comp = self.n_components or min(50, self.output_size[0])
            self._reducer = PCA(n_components=n_comp, random_state=self.random_state)
        elif self.feature_reduction.lower() == 'kmeans':
            n_comp = self.n_components or min(50, self.output_size[0])
            self._reducer = KMeans(n_clusters=n_comp, random_state=self.random_state, n_init=10)
        else:
            raise ValueError(f"Unknown feature reduction method: {self.feature_reduction}")

    def _compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix based on the specified method.

        Parameters:
        -----------
        X : ndarray
            Input feature matrix (samples x features)

        Returns:
        --------
        gram_matrix : ndarray
            Computed Gram matrix
        """
        if self.gram_method == 'correlation':
            # Compute feature correlation matrix
            gram = np.corrcoef(X.T)  # Features x Features

        elif self.gram_method == 'covariance':
            # Compute feature covariance matrix
            gram = np.cov(X.T)  # Features x Features

        elif self.gram_method == 'inner_product':
            # Direct inner product between samples
            gram = np.dot(X, X.T)  # Samples x Samples

        elif self.gram_method == 'rbf_kernel':
            # RBF kernel matrix
            gamma = self.kernel_params.get('gamma', 1.0 / X.shape[1])
            distances = pdist(X, metric='euclidean')
            distances = squareform(distances)
            gram = np.exp(-gamma * distances ** 2)

        elif self.gram_method == 'polynomial_kernel':
            # Polynomial kernel matrix
            degree = self.kernel_params.get('degree', 2)
            coef0 = self.kernel_params.get('coef0', 1)
            gram = (np.dot(X, X.T) + coef0) ** degree

        else:
            raise ValueError(f"Unknown gram method: {self.gram_method}")

        # Handle NaN values
        gram = np.nan_to_num(gram, nan=0.0, posinf=1.0, neginf=-1.0)

        return gram

    def _gram_to_image(self, gram_matrix: np.ndarray) -> np.ndarray:
        """
        Convert Gram matrix to image format.

        Parameters:
        -----------
        gram_matrix : ndarray
            Gram matrix to convert

        Returns:
        --------
        image : ndarray
            Image representation of the Gram matrix
        """
        # Normalize to [0, 1]
        gram_norm = gram_matrix.copy()
        gram_min, gram_max = gram_norm.min(), gram_norm.max()
        if gram_max > gram_min:
            gram_norm = (gram_norm - gram_min) / (gram_max - gram_min)

        # Resize to target output size
        if gram_norm.shape != self.output_size:
            gram_norm = cv2.resize(gram_norm, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Convert to appropriate image format
        if self.img_format == 'rgb':
            # Create RGB image using different color channels
            # Use the matrix as intensity and create color variations
            image = np.zeros((self.output_size[0], self.output_size[1], 3))

            # Channel 1: Original matrix
            image[:, :, 0] = gram_norm

            # Channel 2: Matrix rotated/transformed
            image[:, :, 1] = np.rot90(gram_norm)

            # Channel 3: Matrix with different transformation
            image[:, :, 2] = np.fliplr(gram_norm)

        elif self.img_format == 'scalar':
            image = gram_norm

        else:  # pytorch format
            if len(gram_norm.shape) == 2:
                image = gram_norm[np.newaxis, :, :]  # Add channel dimension
            else:
                image = np.transpose(gram_norm, (2, 0, 1))  # CHW format

        return image

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
        self : GramMatrixPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting Gram Matrix pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Gram method: {self.gram_method}")
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

        # Apply feature reduction if specified
        if self._reducer is not None:
            if hasattr(self._reducer, 'fit_transform'):
                X_reduced = self._reducer.fit_transform(X_scaled)
            else:
                X_reduced = self._reducer.fit(X_scaled).cluster_centers_
            if self.verbose:
                print(
                    f"Features reduced from {X_scaled.shape[1]} to {X_reduced.shape[1]} using {type(self._reducer).__name__}")
        else:
            X_reduced = X_scaled

        # Compute Gram matrix for the training data
        try:
            gram_matrix = self._compute_gram_matrix(X_reduced)

            # Store the reference Gram matrix
            self._gram_matrices = {
                'reference': gram_matrix,
                'feature_means': np.mean(X_reduced, axis=0),
                'feature_stds': np.std(X_reduced, axis=0)
            }

            self._is_fitted = True
            if self.verbose:
                print(f"Gram matrix computed with shape: {gram_matrix.shape}")
                print("Pipeline fitted successfully!")

        except Exception as e:
            if self.verbose:
                print(f"Error during fitting: {e}")
            raise e

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using Gram matrices.

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
            print(f"Transforming data to images using Gram matrices...")
            print(f"Input shape: {X.shape}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Apply feature reduction if specified
        if self._reducer is not None:
            if hasattr(self._reducer, 'transform'):
                X_reduced = self._reducer.transform(X_scaled)
            else:
                # For KMeans, compute distances to cluster centers
                distances = np.linalg.norm(X_scaled[:, np.newaxis] - self._reducer.cluster_centers_, axis=2)
                X_reduced = 1.0 / (1.0 + distances)  # Convert distances to similarity scores
        else:
            X_reduced = X_scaled

        # Generate images
        images = []

        for i in range(X_reduced.shape[0]):
            if self.gram_method in ['correlation', 'covariance']:
                # For feature-based methods, create a modified version of the reference
                sample = X_reduced[i:i + 1]  # Keep as 2D

                # Create a combined matrix with the sample and reference features
                combined = np.vstack([sample, self._gram_matrices['feature_means'].reshape(1, -1)])
                gram_matrix = self._compute_gram_matrix(combined)

            else:
                # For sample-based methods, compute gram matrix with reference samples
                sample = X_reduced[i:i + 1]

                # Create gram matrix between sample and reference
                if self.gram_method == 'inner_product':
                    gram_matrix = np.dot(sample, sample.T)
                elif self.gram_method == 'rbf_kernel':
                    gamma = self.kernel_params.get('gamma', 1.0 / sample.shape[1])
                    gram_matrix = np.exp(-gamma * np.linalg.norm(sample) ** 2)
                    gram_matrix = np.array([[gram_matrix]])
                elif self.gram_method == 'polynomial_kernel':
                    degree = self.kernel_params.get('degree', 2)
                    coef0 = self.kernel_params.get('coef0', 1)
                    gram_matrix = (np.dot(sample, sample.T) + coef0) ** degree

                # Ensure it's at least 2x2 for image conversion
                if gram_matrix.shape[0] == 1:
                    gram_matrix = np.tile(gram_matrix, (2, 2))

            # Convert to image
            image = self._gram_to_image(gram_matrix)
            images.append(image)

        X_img = np.array(images)

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'scaler': self.scaler,
            'gram_method': self.gram_method,
            'feature_reduction': self.feature_reduction,
            'n_components': self.n_components,
            'kernel_params': self.kernel_params,
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
        if 'feature_reduction' in params or 'n_components' in params:
            self._setup_reducer()

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
