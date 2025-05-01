"""
This script defines a custom data generator for leaf images and their color
histograms.
The generator yields batches of images and their corresponding labels,
allowing for efficient data loading and preprocessing during model training.

Dependencies:
    - os
    - cv2
    - numpy
    - pandas
    - tensorflow.keras.utils
    - Transformation (custom module for image transformations)
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from Transformation import apply_transformations


class LeafDataGenerator(Sequence):
    """
    Custom data generator for leaf images and their color histograms.
    This generator yields batches of images and their corresponding labels.

    @ivar samples: List of tuples containing image paths and class names
    @type samples: list
    @ivar class_names: List of class names
    @type class_names: list
    @ivar num_classes: Number of classes
    @type num_classes: int
    @ivar batch_size: Size of each batch
    @type batch_size: int
    @ivar image_size: Size of the images
    @type image_size: tuple
    @ivar shuffle: Whether to shuffle the data at the end of each epoch
    @type shuffle: bool
    @ivar mode: Mode of the generator (train/validation)
    @type mode: str
    """
    def __init__(self, samples, class_names, batch_size=32,
                 image_size=(256, 256), shuffle=True, mode='train'):
        """
        Initialize the data generator.

        @param samples: List of tuples containing image paths and class names
        @type samples: list
        @param class_names: List of class names
        @type class_names: list
        @param batch_size: Size of each batch
        @type batch_size: int
        @param image_size: Size of the images
        @type image_size: tuple
        @param shuffle: Whether to shuffle the data at the end of each epoch
        @type shuffle: bool
        @param mode: Mode of the generator (train/validation)
        @type mode: str

        @return: None
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        self.samples = samples
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.mode = mode

        self.on_epoch_end()

    def __len__(self):
        """
        Return the number of batches per epoch.

        @return: Number of batches per epoch
        @rtype: int
        """
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch if shuffle is True.
        This method is called at the end of each epoch during training.

        @return: None
        """
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        """
        Generate one batch of data.

        @param idx: Index of the batch
        @type idx: int

        @return: Tuple of (X, y) where X is a list of image batches
        and y is the corresponding labels
        @rtype: tuple
        """
        id_start = idx * self.batch_size
        id_end = (idx + 1) * self.batch_size
        batch_samples = self.samples[id_start:id_end]

        # Prepare output arrays
        image_batches = [[] for _ in range(6)]  # 6 transformed images
        histogram_batch = []
        label_batch = []

        for img_path, class_name in batch_samples:
            image_base = os.path.splitext(os.path.basename(img_path))[0]
            cache_dir = os.path.join(f"cache/{self.mode}", class_name)

            transformed = {}
            all_exist = True

            # Check for cached images
            image_keys = ["Original", "Gaussian Blur", "Mask", "ROI",
                          "Analyze Objects", "Landmark"]
            for key in image_keys:
                cached_path = os.path.join(cache_dir,
                                           f"{image_base}_{key}.jpg")
                if os.path.exists(cached_path):
                    transformed[key] = cv2.imread(cached_path)
                else:
                    all_exist = False

            # Check for cached histogram
            histogram_path = os.path.join(cache_dir, f"{image_base}_hist.csv")
            if os.path.exists(histogram_path):
                hist_df = pd.read_csv(histogram_path)
                transformed["Color_Histogram"] = hist_df
            else:
                all_exist = False

            # If anything is missing, recompute everything
            if not all_exist:
                transformed = apply_transformations(img_path, plot=False)

                # Save images
                os.makedirs(cache_dir, exist_ok=True)
                for key in image_keys:
                    cv2.imwrite(
                        os.path.join(cache_dir, f"{image_base}_{key}.jpg"),
                        transformed[key]
                    )

                # Save histogram
                hist_df = transformed["Color_Histogram"]
                hist_df.to_csv(histogram_path, index=False)

            # Collect transformed images
            image_keys = [k for k in transformed if k != 'Color_Histogram']
            for i, key in enumerate(image_keys):
                img_array = transformed[key]
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array = img_array.astype('float32') / 255.0
                image_batches[i].append(img_array)

            # Collect histogram
            histogram = transformed['Color_Histogram']
            values_only = histogram.iloc[:, 1].values.astype(np.float32)
            histogram_batch.append(values_only.flatten())

            # Collect label
            label_index = self.class_names.index(class_name)
            label_batch.append(to_categorical(label_index,
                                              num_classes=self.num_classes))

        # Stack inputs for Keras
        X = ([np.stack(batch) for batch in image_batches]
             + [np.stack(histogram_batch)])
        y = np.stack(label_batch)

        return tuple(X), y
