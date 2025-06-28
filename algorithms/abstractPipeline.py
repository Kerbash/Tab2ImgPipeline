from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Any, Dict
import os


class BaseDataTransformPipeline(ABC):
    """
    Abstract base class for data transformation pipelines.

    This class defines the common interface that all transformation pipelines
    should implement, allowing for consistent usage across different algorithms.

    Parameters:
    -----------
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Optional[Union[pd.Series, np.ndarray]] = None):
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
        self : BaseDataTransformPipeline
            Returns self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data using the fitted pipeline.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix to transform

        Returns:
        --------
        X_transformed : ndarray
            Transformed data
        """
        pass

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
        """
        Fit the pipeline and transform the data in one step.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector

        Returns:
        --------
        X_transformed : ndarray
            Transformed data
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get pipeline parameters.

        Returns:
        --------
        params : dict
            Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """
        Set pipeline parameters.

        Parameters:
        -----------
        **params : dict
            Parameter names and values to set

        Returns:
        --------
        self : BaseDataTransformPipeline
            Returns self for method chaining
        """
        pass

    def save_output(self, path: str, X_transformed: np.ndarray,
                    y: Optional[Union[pd.Series, np.ndarray]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Save transformed data to disk with metadata.

        This is a generic save method that can be overridden by specific implementations
        for specialized saving (like images for DeepInsight).

        Parameters:
        -----------
        path : str
            Directory path where data will be saved
        X_transformed : ndarray
            Transformed data to save
        y : Series or ndarray, optional
            Target vector (labels)
        metadata : dict, optional
            Additional metadata to save
        """
        os.makedirs(path, exist_ok=True)

        if self.verbose:
            print(f"Saving transformed data to: {path}")

        # Save transformed data
        np.save(os.path.join(path, 'transformed_data.npy'), X_transformed)

        # Create and save key file with metadata
        key_data = []
        for i in range(X_transformed.shape[0]):
            entry = {
                'sample_index': i,
                'label': y[i] if y is not None else None
            }
            if metadata:
                entry.update(metadata)
            key_data.append(entry)

        key_df = pd.DataFrame(key_data)
        key_path = os.path.join(path, 'key.csv')
        key_df.to_csv(key_path, index=False)

        if self.verbose:
            print(f"Saved transformed data shape: {X_transformed.shape}")
            print(f"Key file saved to: {key_path}")
            if y is not None:
                print(f"Label distribution:")
                print(key_df['label'].value_counts())

    def fit_transform_save(self, path: str, X: Union[pd.DataFrame, np.ndarray],
                           y: Optional[Union[pd.Series, np.ndarray]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Fit the pipeline, transform the data, and save the results.

        Parameters:
        -----------
        path : str
            Directory path where data will be saved
        X : DataFrame or ndarray
            Feature matrix
        y : Series or ndarray, optional
            Target vector
        metadata : dict, optional
            Additional metadata to save

        Returns:
        --------
        X_transformed : ndarray
            Transformed data
        """
        X_transformed = self.fit_transform(X, y)
        self.save_output(path, X_transformed, y, metadata)
        return X_transformed

    @property
    def is_fitted(self) -> bool:
        """Check if the pipeline has been fitted."""
        return self._is_fitted

    def _validate_fitted(self):
        """Check if pipeline is fitted, raise error if not."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")

    def _convert_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X.copy()

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class BaseImageTransformPipeline(BaseDataTransformPipeline, ABC):
    """
    Abstract base class specifically for pipelines that transform data to images.

    Extends the base pipeline with image-specific functionality.

    Parameters:
    -----------
    output_size : tuple of int, default=(224, 224)
        The size of the output images (height, width)
    img_format : str, default='rgb'
        Output image format
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(self, output_size: Tuple[int, int] = (224, 224),
                 img_format: str = 'rgb', random_state: int = 42,
                 verbose: bool = True):
        super().__init__(random_state=random_state, verbose=verbose)
        self.output_size = output_size
        self.img_format = img_format

    def save_output(self, path: str, X_transformed: np.ndarray,
                    y: Optional[Union[pd.Series, np.ndarray]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Save transformed images to disk.

        Parameters:
        -----------
        path : str
            Directory path where images will be saved
        X_transformed : ndarray
            Transformed image data
        y : Series or ndarray, optional
            Target vector (labels)
        metadata : dict, optional
            Additional metadata to save
        """
        from PIL import Image
        import json

        os.makedirs(path, exist_ok=True)

        if self.verbose:
            print(f"Saving images to: {path}")

        key_data = []

        # Save individual images
        for i in range(X_transformed.shape[0]):
            # Generate filename
            filename = f"sample_{i:06d}.png"
            filepath = os.path.join(path, filename)

            # Get the image for this sample
            img_data = X_transformed[i]

            # Convert and save image
            img = self._convert_to_pil_image(img_data)
            img.save(filepath)

            # Save numerical array as JSON
            json_filename = f"sample_{i:06d}.json"
            json_filepath = os.path.join(path, json_filename)
            with open(json_filepath, 'w') as f:
                json.dump(img_data.tolist(), f)

            # Add to key data
            entry = {
                'filename': filename,
                'og_index': i,
                'label': y.iloc[i] if y is not None else None
            }
            if metadata:
                entry.update(metadata)
            key_data.append(entry)

        # Save key file
        key_df = pd.DataFrame(key_data)
        key_path = os.path.join(path, 'key.csv')
        key_df.to_csv(key_path, index=False)

        if self.verbose:
            print(f"Saved {len(key_data)} images")
            print(f"Key file saved to: {key_path}")
            if y is not None:
                print(f"Label distribution:")
                print(key_df['label'].value_counts())

    def _convert_to_pil_image(self, img_data: np.ndarray):
        """Convert numpy array to PIL Image."""
        from PIL import Image

        # Handle different image formats
        if self.img_format == 'rgb' and img_data.ndim == 3:
            # RGB format
            if img_data.shape[2] == 3:
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
                return Image.fromarray(img_data, mode='RGB')
            else:
                # Single channel, convert to grayscale
                img_data = img_data.squeeze()

        # Handle grayscale or convert to grayscale
        if img_data.ndim > 2:
            img_data = np.mean(img_data, axis=-1)

        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)

        return Image.fromarray(img_data, mode='L')