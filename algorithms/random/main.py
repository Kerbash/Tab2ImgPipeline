import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Union, Tuple, Any, Dict

# Import the abstract base class
from algorithms.abstractPipeline import BaseImageTransformPipeline


class RandomStackPipeline(BaseImageTransformPipeline):
    """
    A pipeline class for converting tabular data to images using random feature stacking.

    This serves as a control method that randomly arranges features into image format
    without any intelligent dimensionality reduction or spatial relationships.

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

    img_format : str, default='rgb'
        Output image format. Options: 'rgb', 'scalar'

    shuffle_features : bool, default=True
        Whether to shuffle feature order randomly

    pad_mode : str, default='reflect'
        How to handle cases where features don't fill the image perfectly.
        Options: 'reflect', 'constant', 'wrap', 'duplicate'

    pad_value : float, default=0.0
        Value to use for constant padding (only used if pad_mode='constant')

    random_state : int, default=42
        Random state for reproducibility

    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (224, 224),
            scaler: Union[str, Any] = 'standard',
            img_format: str = 'rgb',
            shuffle_features: bool = True,
            pad_mode: str = 'reflect',
            pad_value: float = 0.0,
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

        # RandomStack-specific parameters
        self.scaler = scaler
        self.shuffle_features = shuffle_features
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        # Initialize components
        self._scaler_obj = None
        self._feature_indices = None
        self._total_pixels = output_size[0] * output_size[1]

        # For RGB format, we need 3 channels
        if img_format == 'rgb':
            self._channels = 3
            self._pixels_per_channel = self._total_pixels
        else:
            self._channels = 1
            self._pixels_per_channel = self._total_pixels

        # Setup scaler
        self._setup_scaler()

        # Set random seed
        np.random.seed(self.random_state)

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

    def _create_feature_mapping(self, n_features: int):
        """Create mapping from features to image positions."""
        if self.verbose:
            print(f"Creating feature mapping for {n_features} features")
            print(f"Target image size: {self.output_size}")
            print(f"Total pixels needed: {self._total_pixels}")
            print(f"Channels: {self._channels}")

        # Create feature indices
        if self.shuffle_features:
            self._feature_indices = np.random.permutation(n_features)
        else:
            self._feature_indices = np.arange(n_features)

        # Calculate how many times we need to repeat/pad features
        total_positions_needed = self._total_pixels * self._channels

        if n_features >= total_positions_needed:
            # We have more features than positions, select subset
            self._feature_mapping = self._feature_indices[:total_positions_needed]
        else:
            # We need to repeat/pad features to fill all positions
            self._feature_mapping = self._pad_features(n_features, total_positions_needed)

        if self.verbose:
            print(f"Feature mapping created: {len(self._feature_mapping)} positions filled")

    def _pad_features(self, n_features: int, total_needed: int) -> np.ndarray:
        """Pad features to fill the required positions."""
        mapping = []

        if self.pad_mode == 'reflect':
            # Reflect the feature array
            extended_features = np.concatenate([self._feature_indices, self._feature_indices[::-1]])
            repeats = (total_needed // len(extended_features)) + 1
            full_mapping = np.tile(extended_features, repeats)
            mapping = full_mapping[:total_needed]

        elif self.pad_mode == 'wrap':
            # Wrap around the features
            repeats = (total_needed // n_features) + 1
            full_mapping = np.tile(self._feature_indices, repeats)
            mapping = full_mapping[:total_needed]

        elif self.pad_mode == 'duplicate':
            # Duplicate the last feature
            mapping = np.full(total_needed, self._feature_indices[-1])
            mapping[:n_features] = self._feature_indices

        elif self.pad_mode == 'constant':
            # This will be handled differently - we'll pad with the constant value
            mapping = np.full(total_needed, -1)  # -1 indicates constant padding
            mapping[:n_features] = self._feature_indices

        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

        return mapping

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
        self : RandomStackPipeline
            Returns self for method chaining
        """
        if self.verbose:
            print(f"Fitting RandomStack pipeline...")
            print(f"Input shape: {X.shape}")
            print(f"Output size: {self.output_size}")
            print(f"Image format: {self.img_format}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)
        n_features = X_array.shape[1]

        # Scale the data
        if self._scaler_obj is not None:
            self._scaler_obj.fit(X_array)
            if self.verbose:
                print(f"Scaler fitted: {type(self._scaler_obj).__name__}")
        else:
            if self.verbose:
                print("No scaling will be applied")

        # Create feature mapping
        self._create_feature_mapping(n_features)

        self._is_fitted = True
        if self.verbose:
            print("Pipeline fitted successfully!")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data to images using random stacking.

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
            print(f"Transforming data to images using random stacking...")
            print(f"Input shape: {X.shape}")

        # Convert to numpy if pandas
        X_array = self._convert_input(X)
        n_samples, n_features = X_array.shape

        # Scale the data
        if self._scaler_obj is not None:
            X_scaled = self._scaler_obj.transform(X_array)
        else:
            X_scaled = X_array

        # Initialize output array
        if self.img_format == 'rgb':
            X_img = np.zeros((n_samples, self.output_size[0], self.output_size[1], 3))
        else:
            X_img = np.zeros((n_samples, self.output_size[0], self.output_size[1]))

        # Transform each sample
        for i in range(n_samples):
            X_img[i] = self._sample_to_image(X_scaled[i])

        if self.verbose:
            print(f"Output image shape: {X_img.shape}")

        return X_img

    def _sample_to_image(self, sample: np.ndarray) -> np.ndarray:
        """Convert a single sample to image format."""
        if self.img_format == 'rgb':
            # Create RGB image
            img = np.zeros((self.output_size[0], self.output_size[1], 3))

            # Fill each channel
            for channel in range(3):
                start_idx = channel * self._pixels_per_channel
                end_idx = (channel + 1) * self._pixels_per_channel
                channel_mapping = self._feature_mapping[start_idx:end_idx]
                channel_data = self._get_channel_data(sample, channel_mapping)
                img[:, :, channel] = channel_data.reshape(self.output_size)

        else:
            # Create grayscale image
            channel_data = self._get_channel_data(sample, self._feature_mapping)
            img = channel_data.reshape(self.output_size)

        return img

    def _get_channel_data(self, sample: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """Get data for a single channel based on feature mapping."""
        channel_data = np.zeros(len(mapping))

        for i, feature_idx in enumerate(mapping):
            if feature_idx == -1:  # Constant padding
                channel_data[i] = self.pad_value
            else:
                channel_data[i] = sample[feature_idx]

        return channel_data

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            'output_size': self.output_size,
            'scaler': self.scaler,
            'img_format': self.img_format,
            'shuffle_features': self.shuffle_features,
            'pad_mode': self.pad_mode,
            'pad_value': self.pad_value,
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

        # Update derived attributes
        if 'output_size' in params:
            self._total_pixels = self.output_size[0] * self.output_size[1]
            if self.img_format == 'rgb':
                self._pixels_per_channel = self._total_pixels

        if 'img_format' in params:
            if self.img_format == 'rgb':
                self._channels = 3
                self._pixels_per_channel = self._total_pixels
            else:
                self._channels = 1
                self._pixels_per_channel = self._total_pixels

        # Update parent class attributes
        if 'output_size' in params:
            super().__setattr__('output_size', params['output_size'])
        if 'img_format' in params:
            super().__setattr__('img_format', params['img_format'])
        if 'random_state' in params:
            super().__setattr__('random_state', params['random_state'])
            np.random.seed(self.random_state)
        if 'verbose' in params:
            super().__setattr__('verbose', params['verbose'])

        # Reset fitted state
        self._is_fitted = False
        return self