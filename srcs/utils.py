"""
This script provides utility functions for image processing and file handling.

Dependencies:
    - os
    - cv2
    - numpy
    - PIL (Pillow)
"""

import os
import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image):
    """
    Convert PIL image to OpenCV format.

    @param image: PIL Image
    @type image: PIL.Image.Image

    @return: OpenCV image
    @rtype: numpy.ndarray
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    """
    Convert OpenCV image to PIL format.

    @param image: OpenCV image
    @type image: numpy.ndarray

    @return: PIL Image
    @rtype: PIL.Image.Image
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def is_image_file(path):
    """
    Check if the given path is a valid image file.

    @param path: Path to the file
    @type path: str

    @return: True if the file is a valid image, False otherwise
    @rtype: bool
    """
    image_extensions = ('.jpg', '.jpeg', '.png')
    return os.path.isfile(path) and path.lower().endswith(image_extensions)


def is_directory(path):
    """
    Check if the given path is a directory.

    @param path: Path to the directory
    @type path: str

    @return: True if the path is a directory, False otherwise
    @rtype: bool
    """
    return os.path.isdir(path)
