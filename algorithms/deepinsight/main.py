import numpy as np
import pandas as pd
import pyDeepInsight
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from typing import Optional, Union, Tuple, Any, Dict

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class DeepInsightPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using DeepInsight methodology.

    Parameters:
    -----------
    feature_extractor : str or object, default='tsne'
        The dimensionality reduction method to use. Options:
        - 'tsne': t-SNE with default parameters
        - 'pca': PCA with default parameters
        - 'umap': UMAP with default parameters
        - Custom sklearn-compatible object with fit_transform method

    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)

    discretization : str, default='bin'
        The discretization method. Options: 'bin', 'lsa', 'qtb', 'sla', 'ags'

    scaler : str or object, default='standard'
        The scaling method for features. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        - Custom sklearn-compatible scaler object

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'scalar', 'pytorch'

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            feature_extractor: Union[str, Any] = 'tsne',
            output_size: Tuple[int, int] = (224, 224),
            discretization: str = 'bin',
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

        # DeepInsight-specific parameters
        self.feature_extractor = feature_extractor
        self.discretization = discretization
        self.scaler = scaler

        # Initialize components
        self._scaler_obj = None
        self._transformer = None
        self._feature_extractor_obj = None

        # Setup feature extractor and scaler
        self._setup_feature_extractor()
        self._setup_scaler()

    def _setup_feature_extractor(self):
        """Setup the feature extractor object."""
        if isinstance(self.feature_extractor, str):
            if self.feature_extractor.lower() == 'tsne':
                self._feature_extractor_obj = TSNE(
                    n_components=2,
                    random_state=self.random_state,
                    perplexity=min(30, max(5, self.random_state // 10))  # Adaptive perplexity
                )
            elif self.feature_extractor.lower() == 'pca':
                self._feature_extractor_obj = PCA(
                    n_components=2,
                    random_state=self.random_state
                )
            elif self.feature_extractor.lower() == 'umap':
                self._feature_extractor_obj = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unknown feature extractor: {self.feature_extractor}")
        else:
            self._feature_extractor_obj = self.feature_extractor

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
        Fit the pipeline to the training data.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector (not used in transformation but kept for sklearn compatibility)

        Returns:
        --------
        self : DeepInsightPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting DeepInsight pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Feature extractor: {type(self._feature_extractor_obj).__name__}")
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

        # Adjust perplexity for t-SNE if needed
        if isinstance(self._feature_extractor_obj, TSNE):
            n_samples = X_scaled.shape[0]
            if hasattr(self._feature_extractor_obj, 'perplexity'):
                perplexity = min(self._feature_extractor_obj.perplexity, (n_samples - 1) // 3)
                if perplexity != self._feature_extractor_obj.perplexity:
                    if self.verbose:
                        print(
                            f"Adjusting t-SNE perplexity from {self._feature_extractor_obj.perplexity} to {perplexity}")
                    self._feature_extractor_obj.perplexity = perplexity

        # Create ImageTransformer
        self._transformer = pyDeepInsight.ImageTransformer(
            feature_extractor=self._feature_extractor_obj,
            discretization=self.discretization,
            pixels=self.output_size
        )

        # Fit the transformer
        try:
            self._transformer.fit(X_scaled)
            self._is_fitted = True
            if self.verbose:
                print("Pipeline fitted successfully!")
        except Exception as e:
            if self.verbose:
                print(f"Error during fitting: {e}")
            raise e

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

        # Convert to numpy if pandas
        X_array = self._convert_input(X)

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Transform to images
        X_img = self._transformer.transform(X_scaled, img_format=self.img_format)

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'feature_extractor': self.feature_extractor,
            'output_size': self.output_size,
            'discretization': self.discretization,
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
        if 'feature_extractor' in params:
            self._setup_feature_extractor()
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
