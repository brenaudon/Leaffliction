"""
This script defines a function to create a TensorFlow dataset
for training a model on prepared leaf images.
It reads images and histograms from a prepared dataset,
and returns a TensorFlow dataset object.

Dependencies:
    - os
    - cv2
    - json
    - numpy
    - tensorflow as tf
"""

import os
import cv2
import json
import numpy as np
import tensorflow as tf

PREP_DIR = "cache/prepared"
IMG_KEYS = ["Original", "Gaussian Blur", "Mask",
            "ROI", "Analyze Objects", "Landmark"]
HIST_SIZE = 2304
BATCH = 16


def make_dataset(samples, shuffle=True):
    """
    Create a TensorFlow dataset from the list of samples.

    @param samples: List of tuples containing image paths and class names
    @type samples: list
    @param batch_size: Size of each batch
    @type batch_size: int
    @param shuffle: Whether to shuffle the data at the end of each epoch
    @type shuffle: bool

    @return: TensorFlow dataset
    @rtype: tf.data.Dataset
    """
    with open(os.path.join(PREP_DIR, "manifest.json")) as f:
        manifest = json.load(f)
        rel_to_lbl = {m["rel"]: m["label"] for m in manifest}
        num_classes = max(rel_to_lbl.values()) + 1

    def _safe_read_img(path):
        """
        Read an image from the given path and preprocess it.
        This function ensures the image is resized to 256x256 pixels
        and converted to a float32 array with values in the range [0, 1].

        @param path: Path to the image file
        @type path: str

        @return: Preprocessed image as a NumPy array
        @rtype: numpy.ndarray
        """
        data = tf.io.read_file(path)
        img = tf.image.decode_jpeg(data, channels=3)
        arr = img.numpy()

        # force size 256×256
        if arr.shape[:2] != (256, 256):        # compare (H,W)
            arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)

        return arr.astype("float32") / 255.0

    def _load_one_sample(rel_bytes):
        """
        Load a single sample from the prepared dataset.
        This function decodes the relative path, constructs the full path,
        and loads the corresponding images and histogram.

        @param rel_bytes: Relative path to the sample as bytes
        @type rel_bytes: bytes

        @return: Tuple of images and one-hot encoded label
        @rtype: tuple
        """
        rel = rel_bytes.numpy().decode("utf-8")
        # rel of shape images/{class_name}/{name}.jpg
        class_name = rel.split("/")[1]
        rel_name = os.path.splitext(os.path.basename(rel))[0]
        base = os.path.join(PREP_DIR, class_name, rel_name)

        imgs = [_safe_read_img(f"{base}_{k}.jpg") for k in IMG_KEYS]

        # histogram
        hist = np.load(f"{base}_hist.npy").astype("float32")

        # label
        rel = os.path.join(class_name, rel_name)
        label_idx = rel_to_lbl[rel]
        # one-hot (bit of the label to 1 and rest to 0)
        label = np.eye(num_classes, dtype="float32")[label_idx]

        return imgs + [hist, label]

    def tf_loader(rel):
        """
        TensorFlow function to load one sample from the dataset.

        @param rel: Relative path to the sample as a TensorFlow string tensor
        @type rel: tf.Tensor

        @return: Tuple of images and one-hot encoded label
        @rtype: tuple
        """
        out_tensors = tf.py_function(_load_one_sample,
                                     inp=[rel],
                                     Tout=[tf.float32] * 8)

        imgs = out_tensors[:6]
        hist = out_tensors[6]
        label = out_tensors[7]

        # set static shapes
        for img in imgs:
            img.set_shape([256, 256, 3])
        hist.set_shape([HIST_SIZE])
        label.set_shape([num_classes])

        features = tuple(imgs + [hist])
        return features, label

    rel_list = [s[0] for s in samples]
    rel_ds = tf.data.Dataset.from_tensor_slices([s.encode() for s in rel_list])
    if shuffle:
        rel_ds = rel_ds.shuffle(len(rel_list))
    ds = rel_ds.map(tf_loader, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
