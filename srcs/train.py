"""
This script defines a Keras model for classifying leaf images.
It uses a shared CNN encoder for processing multiple images and a histogram,
and combines their features for classification.

Dependencies:
    - os
    - sys
    - random
    - tensorflow.keras.layers
    - tensorflow.keras.Input
    - tensorflow.keras.Model
    - generator (custom module for data generation)
    - save_zip (custom module for saving model and zip it with cache)
"""

import os
import sys
import random
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from generator import LeafDataGenerator
from save_zip import save_model_and_cache


def create_encoder():
    """
    Create a shared CNN encoder for the 6 images.

    @return: Keras Model
    @rtype: keras.Model
    """
    input_img = Input(shape=(256, 256, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    return Model(inputs=input_img, outputs=x)


def create_model(num_classes):
    """
    Create a model that takes 6 images and a histogram as input
    and outputs class probabilities.

    @param num_classes: Number of classes for classification
    @type num_classes: int

    @return: Keras Model
    @rtype: keras.Model
    """
    # Shared encoder for 6 images
    encoder = create_encoder()

    # Inputs for the 6 images
    image_inputs = [Input(shape=(256, 256, 3)) for _ in range(6)]

    # Apply shared encoder to all 6 images
    image_features = [encoder(img) for img in image_inputs]

    # Histogram input and MLP encoder
    hist_input = Input(shape=(2304,))
    h = layers.Dense(64, activation='relu')(hist_input)
    h = layers.Dense(32, activation='relu')(h)
    hist_features = h

    # Fusion: concatenate all features
    all_features = layers.Concatenate()(image_features + [hist_features])

    # Dense classification head
    x = layers.Dense(256, activation='relu')(all_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Define final model
    model = Model(inputs=image_inputs + [hist_input], outputs=output)

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    """
    Main function to train the model.
    """
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_directory>")
        return

    base_dir = sys.argv[1]

    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        return

    num_classes = len(os.listdir(base_dir))
    model = create_model(num_classes)
    # Gather all samples
    class_names = sorted(os.listdir(base_dir))
    all_samples = []

    # save class names to a file (create if not exists)
    if not os.path.exists("model"):
        os.makedirs("model")
    with open("model/class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_samples.append((os.path.join(class_path, fname),
                                    class_name))

    # Deterministic shuffle + split
    random.seed(42)
    random.shuffle(all_samples)

    train_samples = all_samples[:int(0.9 * len(all_samples))]
    val_samples = all_samples[int(0.9 * len(all_samples)):]

    # Create generators using pre-split data
    train_gen = LeafDataGenerator(train_samples, class_names,
                                  batch_size=8, mode='train')
    val_gen = LeafDataGenerator(val_samples, class_names,
                                batch_size=8, mode='validation')

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=2,
                                   restore_best_weights=True,
                                   start_from_epoch=3)

    model.fit(train_gen,
              validation_data=val_gen,
              epochs=10,
              callbacks=[early_stopping])

    save_model_and_cache(model)


if __name__ == "__main__":
    main()
