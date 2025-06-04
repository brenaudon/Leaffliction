"""
This script predicts the class of an image or a batch of images using
a pre-trained Keras model. It applies necessary transformations,
preprocesses the images, and uses the model to make predictions.

Dependencies:
    - os
    - sys
    - cv2
    - numpy
    - matplotlib.pyplot
    - tensorflow.keras.models
    - generator (custom module for transformations)
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Transformation import apply_transformations


def load_class_names(model_dir="model"):
    """
    Load class names from a file in the specified directory.

    @param model_dir: Directory where the class names file is located.
    @type model_dir: str

    @return: List of class names.
    @rtype: list
    """
    classes_path = os.path.join(model_dir, "class_names.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        raise FileNotFoundError(f"Class names file not found in {model_dir}")


def preprocess_image(img):
    """
    Preprocess the image for model input.
    This function converts the image to RGB, resizes it to 256x256,
    and normalizes pixel values to the range [0, 1].

    @param img: Input image in BGR format (as read by OpenCV).
    @type img: numpy.ndarray

    @return: Preprocessed image ready for model input.
    @rtype: numpy.ndarray
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    return img


def show_result(original, masked, pred_label, confidence):
    """
    This function displays the original and the masked images side by side,
    along with the predicted class label and confidence score in a banner below

    @param original: Original image before processing.
    @type original: numpy.ndarray
    @param masked: Image with mask applied.
    @type masked: numpy.ndarray
    @param pred_label: Predicted class label.
    @type pred_label: str
    @param confidence: Confidence score of the prediction.
    @type confidence: float

    @return: None
    """

    # Convert BGR→RGB for display
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    # Basic figure with black background
    plt.figure(figsize=(5.5, 4.5), facecolor="black")
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])

    # original
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(original)
    ax0.axis("off")
    # masked leaf on white
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(masked)
    ax1.axis("off")

    # Bottom banner
    ax2 = plt.subplot(gs[1, :])
    ax2.axis("off")
    ax2.set_facecolor("black")

    ax2.text(0.5, 0.80, "===      DL classification      ===",
             ha="center", va="center",
             fontsize=14, fontweight="bold", color="white")

    ax2.text(0.5, 0.30,
             f"Class predicted :  {pred_label}",
             ha="center", va="center",
             fontsize=12, color="#5fc46a")

    # confidence
    ax2.text(0.98, 0.05, f"Confidence: {confidence:.1%}",
             ha="right", va="bottom",
             fontsize=10, color="white")

    plt.tight_layout(pad=0.5)
    plt.show()


def predict_single(image_path, model, class_names, show_plot=True):
    """
    Predict the class of a single image using the pre-trained model.
    This function applies necessary transformations to the image,
    preprocesses it, and uses the model to predict the class.

    @param image_path: Path to the image file.
    @type image_path: str
    @param model: Pre-trained Keras model for classification.
    @type model: keras.Model
    @param class_names: List of class names corresponding to model output.
    @type class_names: list
    @param show_plot: Whether to display the prediction result.
    @type show_plot: bool

    @return: Tuple containing predicted class label and confidence score.
    @rtype: tuple (str, float)
    """
    # Load and transform image
    transformed = apply_transformations(image_path, plot=False)

    image_keys = ["Original", "Gaussian Blur", "Mask", "ROI",
                  "Analyze Objects", "Landmark"]
    images = [preprocess_image(transformed[key]) for key in image_keys]
    # Expand because Keras models expect batched input
    images = [np.expand_dims(img, axis=0) for img in images]

    # Process histogram
    histogram = transformed["Color_Histogram"]
    values = histogram.iloc[:, 1].values.astype(np.float32)
    histogram_input = np.expand_dims(values, axis=0)

    # Predict
    prediction = model.predict(images + [histogram_input])  # list of 7 inputs
    predicted_index = np.argmax(prediction[0])
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    if show_plot:
        show_result(transformed["Original"], transformed["Mask"],
                    predicted_label, confidence)

    return predicted_label, confidence


def evaluate_list(paths_list, model, class_names):
    """
    Evaluate all images in a directory and print predictions.
    This function walks through the directory, finds all images,
    applies the model to each image, and prints the predicted class
    along with the true class (based on the parent directory name).

    @param path_list: List of image paths.
    @type path_list: list
    @param model: Pre-trained Keras model for classification.
    @type model: keras.Model
    @param class_names: List of class names corresponding to model output.
    @type class_names: list

    @return: None
    """
    total = len(paths_list)
    correct = 0
    skipped = 0
    for i, path in enumerate(paths_list):
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Skipping.")
            skipped += 1
            continue

        true_class = os.path.basename(os.path.dirname(path))

        pred_class, conf = predict_single(path, model, class_names,
                                          show_plot=False)
        is_ok = pred_class == true_class
        emoji = "✔️" if is_ok else "✖️"
        if is_ok:
            correct += 1

        print(f"{i - skipped}/{total - skipped} | "
              f"{true_class:25s} | "
              f"{pred_class:25s} | "
              f"{conf:6.2%} | {emoji}")

    acc = correct / (total - skipped)
    print("-" * 80)
    print(f"Accuracy on {total} images: {acc:.2%} ({correct}/{total})")


def main():
    """
    Main function to predict class thanks to previously trained model.
    """
    if len(sys.argv) < 2:
        print("Usage: python predict.py "
              "<image_path>|<directory_path>|<txt_file_with_image_paths>")
        sys.exit(1)

    path = sys.argv[1]
    model = load_model("model/model.keras")   # loaded once
    class_names = load_class_names("model")

    image_extensions = (".jpg", ".jpeg", ".png")

    if os.path.isfile(path) and path.lower().endswith(image_extensions):
        predict_single(path, model, class_names, show_plot=True)
    elif os.path.isfile(path) and path.lower().endswith(".txt"):
        with open(path, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
            evaluate_list(image_paths, model, class_names)
    elif os.path.isdir(path):
        # Collect all image files in the directory
        image_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        evaluate_list(image_paths, model, class_names)
    else:
        print("Provided path is neither a file nor a directory.")


if __name__ == "__main__":
    main()
