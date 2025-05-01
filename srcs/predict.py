import sys
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from generator import apply_transformations  # Reuse your function

def load_class_names(model_dir="model"):
    # Optionally load from saved list
    classes_path = os.path.join(model_dir, "class_names.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    return ["Class 0", "Class 1", "Class 2", "Class 3"]  # Default fallback

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    return img

def predict(image_path):
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

    # Load model and class names
    model = load_model("model/model.h5")
    class_names = load_class_names("model")

    # Predict
    prediction = model.predict(images + [histogram_input])  # list of 7 inputs
    predicted_index = np.argmax(prediction[0])
    predicted_label = class_names[predicted_index]

    print(f"Predicted class: {predicted_label} (confidence: {prediction[0][predicted_index]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
