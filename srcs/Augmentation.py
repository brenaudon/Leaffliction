"""
This script applies various augmentations to images in a specified directory
or a single image file. The augmentations include flipping, blurring,
brightness adjustment, cropping, and distortions. The script also
balances the number of images across different classes in a directory.

Dependencies:
    - os
    - sys
    - cv2
    - random
    - numpy
    - tqdm
    - albumentations
    - matplotlib
    - PIL (Pillow)
    - Distribution (custom module)
    - utils (custom module)
"""

import os
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from Distribution import count_images
from utils import pil_to_cv2, cv2_to_pil, is_image_file, is_directory


def augment_image(image_path, number_of_augments=6, display=False):
    """
    Apply augmentations to an image and save the results.

    @param image_path: Path to the image file
    @type image_path: str
    @param number_of_augments: Number of augmentations to apply
    @type number_of_augments: int
    @param display: Whether to display the augmented images or not
    @type display: bool

    @return: None
    """
    if number_of_augments < 1 or number_of_augments > 6:
        raise ValueError("number_of_augments must be between 1 and 6")

    image_path = os.path.abspath(image_path)
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    image_root = os.path.dirname(image_path)

    # Load with Pillow, convert to RGB
    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)

    # Augmentations
    crop_factor = np.random.uniform(0.8, 0.95)
    h, w = np_image.shape[:2]

    def blur(img):
        """
        Apply Gaussian blur to the image.

        @param img: Image to blur
        @type img: PIL.Image.Image

        @return: Blurred image
        @rtype: PIL.Image.Image
        """
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))

    def flip(img):
        """
        Apply horizontal flip to the image.

        @param img: Image to flip
        @type img: PIL.Image.Image

        @return: Flipped image
        @rtype: PIL.Image.Image
        """
        return ImageOps.mirror(img)

    def brightness(img):
        """
        Apply random brightness adjustment to the image.

        @param img: Image to adjust
        @type img: PIL.Image.Image

        @return: Brightness adjusted image
        @rtype: PIL.Image.Image
        """
        return cv2_to_pil(
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0, p=1.0)
            (image=pil_to_cv2(img))["image"]
        )

    def crop(img):
        """
        Apply random crop to the image.

        @param img: Image to crop
        @type img: PIL.Image.Image

        @return: Cropped image
        @rtype: PIL.Image.Image
        """
        return cv2_to_pil(
            A.RandomCrop(height=int(crop_factor * h),
                         width=int(crop_factor * w))(
                image=pil_to_cv2(img))["image"])

    def distortion(img):
        """
        Apply elastic distortion to the image.

        @param img: Image to distort
        @type img: PIL.Image.Image

        @return: Distorted image
        @rtype: PIL.Image.Image
        """
        return cv2_to_pil(
            A.Compose([
                A.PadIfNeeded(min_height=int(h * 1.5), min_width=int(w * 1.5),
                              border_mode=cv2.BORDER_REPLICATE),
                A.ElasticTransform(alpha=100, sigma=5, p=1.0),
                A.CenterCrop(height=h, width=w)
            ])(image=pil_to_cv2(img))["image"])

    def grid_distortion(img):
        """
        Apply grid distortion to the image.

        @param img: Image to distort
        @type img: PIL.Image.Image

        @return: Distorted image
        @rtype: PIL.Image.Image
        """
        return cv2_to_pil(
            A.GridDistortion(num_steps=10, distort_limit=0.5, p=1.0)(
                image=pil_to_cv2(img))["image"])

    augmentation_funcs = {
        "Flip": flip,
        "Blur": blur,
        "Brightness": brightness,
        "Crop": crop,
        "Distortion": distortion,
        "GridDistortion": grid_distortion
    }

    augmentation_funcs_copy = augmentation_funcs.copy()

    # Check if augmentation is already existing for this image
    for aug_name in augmentation_funcs_copy.keys():
        if any(f.startswith(f"{name}_{aug_name}")
               for f in os.listdir(image_root)):
            del augmentation_funcs[aug_name]

    number_of_augments = min(number_of_augments, len(augmentation_funcs))

    # Randomly pick N augmentation types
    chosen = random.sample(list(augmentation_funcs.items()),
                           number_of_augments)

    for aug_name, aug_fn in chosen:
        aug_img = aug_fn(image)
        out_name = f"{name}_{aug_name}{ext}"
        out_path = os.path.join(image_root, out_name)
        aug_img.save(out_path)
        if display:
            print(f"[✓] Saved: {out_path}")

    if display:
        plot_augmentation(image_path, augmentation_funcs_copy)


def plot_augmentation(image_path, augmentations):
    """
    Plot the original and augmented images.

    @param image_path: Path to the original image
    @type image_path: str
    @param augmentations: Dictionary of augmentations
    @type augmentations: dict

    @return: None
    """
    # Display original and augmented images
    basename = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Augmentations for: {basename}", fontsize=16)

    axs = axs.ravel()

    # Original image
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis('off')

    # Augmented images
    for idx, (aug_name, aug_fn) in enumerate(augmentations.items(), start=1):
        name, ext = os.path.splitext(basename)
        aug_img = Image.open(os.path.join(
            os.path.dirname(image_path),
            f"{name}_{aug_name}{ext}"
        )).convert("RGB")
        axs[idx].imshow(aug_img)
        axs[idx].set_title(aug_name)
        axs[idx].axis('off')

    # Hide any unused subplots
    for ax in axs[len(augmentations) + 1:]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def process_directory(input_root):
    """
    Count how much images must be created and
    recursively process all images in a directory.

    @param input_root: Path to the root directory
    @type input_root: str

    @return: None
    """
    image_count = count_images(input_root)

    # Find the class with the minimum number of images
    min_class = min(image_count, key=image_count.get)

    # Compute the goal number of images for each class
    # minimum number of images * 7 as we create 6 augmentations
    goal_count = image_count[min_class] * 7

    num_classes = len(image_count.keys())
    i = 1

    for class_dir in image_count.keys():
        full_class_path = os.path.join(input_root, class_dir)
        # exclude possible augmented images
        current_images = [
            f for f in os.listdir(full_class_path)
            if is_image_file(os.path.join(full_class_path, f)) and '_' not in f
        ]

        count = len(current_images)

        num_needed = goal_count - count
        if num_needed <= 0:
            print(f"[✓] {class_dir} is already balanced.")
            continue

        print(f"[→] Augmenting {class_dir} to reach {goal_count} images"
              f" (current images: {count}) | Class {i}/{num_classes}")

        num_augment_by_img = num_needed // count

        if num_augment_by_img != 0:
            print(f"[→] Each image will be augmented "
                  f"{num_augment_by_img} times.")

            for img_file in tqdm(current_images):
                img_path = os.path.join(full_class_path, img_file)
                augment_image(img_path,
                              number_of_augments=num_augment_by_img)

        # Augment a few more images to reach the goal
        # Could do directly with while but no tqdm progress bar then
        for _ in tqdm(range(goal_count -
                            (count + count * num_augment_by_img))):
            img_file = random.choice(current_images)
            img_path = os.path.join(full_class_path, img_file)
            augment_image(img_path, number_of_augments=1)

        # Ensure we reach the goal count
        # (in case same augmentation was applied twice on the same image)
        while len(os.listdir(full_class_path)) < goal_count:
            img_file = random.choice(current_images)
            img_path = os.path.join(full_class_path, img_file)
            augment_image(img_path, number_of_augments=1)

        i += 1


def main():
    """
    Main function to handle command line arguments and process images.

    @return: None
    """
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py "
              "<path_to_image | path_to_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if is_directory(input_path):
        process_directory(input_path)
    elif is_image_file(input_path):
        augment_image(input_path, display=True)
    else:
        print(f"Error: {input_path} is not a valid image file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
