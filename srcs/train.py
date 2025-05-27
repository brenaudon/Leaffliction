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
import random
import argparse
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from generator import LeafDataGenerator
from save_zip import zip_model_and_cache


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


def get_args():
    ap = argparse.ArgumentParser(description="Train or resume training"
                                             " of the CNN.")
    ap.add_argument("-d", "--data-dir", required=True,
                    help="directory containing one sub-folder per class")
    ap.add_argument("-m", "--model",
                    help="existing .keras file to resume training from")
    return ap.parse_args()


def find_file_with_basename(directory, basename, class_name=None):
    """
    Find a file in the given directory with the specified basename.

    @param directory: Directory to search in
    @type directory: str
    @param basename: Basename of the file to find
    @type basename: str

    @return: Full path to the file if found, else None
    @rtype: str or None
    """
    for root, _, files in os.walk(directory):
        if class_name and not os.path.basename(root) == class_name:
            continue
        for file in files:
            if os.path.splitext(file)[0] == basename:
                return os.path.join(root, file)
    return None


def get_train_val_split(data_dir, class_names, create_split=True):
    """
    Get training and validation samples from the data directory.
    This function gathers all samples from the data directory,
    and if `create_split` is True, it creates a new train/validation split.
    If `create_split` is False, it attempts to load existing validation samples
    from a cache directory.

    @param data_dir: Directory containing class sub-folders
    @type data_dir: str
    @param class_names: List of class names (sub-folder names)
    @type class_names: list
    @param create_split: Whether to create a new train/validation split
    @type create_split: bool

    @return: Tuple of (train_samples, val_samples)
    @rtype: tuple
    """
    # Gather all samples from the data directory
    all_samples = []

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_samples.append((os.path.join(class_path, fname),
                                    class_name))

    val_samples = []
    train_samples = []
    if not create_split:
        # If a model is provided, have to keep the same train/validation split
        # Load validation samples from cache/validation
        validation_cache_dir = "cache/validation"
        if not os.path.exists(validation_cache_dir):
            create_split = True
        else:
            for class_name in class_names:
                class_path = os.path.join(validation_cache_dir, class_name)
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path_in_data = (
                            find_file_with_basename(data_dir,
                                                    os.path.splitext(fname)[0],
                                                    class_name))
                        val_samples.append((file_path_in_data, class_name))
            # train_samples is all_samples minus val_samples
            train_samples = [sample for sample in all_samples
                             if sample not in val_samples]

            if val_samples[0] not in all_samples:
                print(f"{val_samples[0]} not in all_samples")
            print(f"Loaded {len(val_samples)} validation samples "
                  f"from {validation_cache_dir}")
            print(f"Loaded {len(train_samples)} training samples "
                  f"from {data_dir}")

    # Create train/validation split
    if create_split:
        # Deterministic shuffle + split
        random.seed(42)
        random.shuffle(all_samples)

        train_samples = all_samples[:int(0.9 * len(all_samples))]
        val_samples = all_samples[int(0.9 * len(all_samples)):]

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise SystemExit("Not enough samples for training or validation. "
                         "Ensure the data directory contains images.")

    return train_samples, val_samples


def main():
    """
    Main function to train the model.
    """
    args = get_args()
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    model_dir = "model"
    create_split = True if not args.model else False

    num_classes = len(os.listdir(data_dir))
    class_names = sorted(os.listdir(data_dir))

    # save class names to a file (create if not exists)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    # Get training and validation samples
    train_samples, val_samples = get_train_val_split(
        data_dir, class_names, create_split=create_split)

    # Create generators using pre-split data
    train_gen = LeafDataGenerator(train_samples, class_names,
                                  batch_size=2, mode='train')
    val_gen = LeafDataGenerator(val_samples, class_names,
                                batch_size=2, mode='validation')

    # Create or load model
    if args.model:
        print(f"Loading model from {args.model}")
        model = load_model(args.model)
        if model.output_shape[-1] != num_classes:
            raise SystemExit("Class count in model "
                             f"({model.output_shape[-1]}) doesn't match "
                             f"folders ({num_classes}).")
    else:
        print("Creating new model")
        model = create_model(num_classes)

    model.fit(train_gen, validation_data=val_gen, epochs=1)

    total_epochs = 15

    for epoch in range(1, total_epochs):
        user_input = input(f"Epoch {epoch + 1}/{total_epochs}."
                           f" Continue training? (y/n): ")

        if user_input.lower() != 'y':
            print("Training stopped by user.")
            user_input = input("Save last model? (y/n): ")
            if user_input.lower() == 'y':
                print("Saving model...")
                model.save(model_dir + "/model.keras")
            else:
                print("Model not saved.")
            break

        # Save model
        print("Saving last model...")
        model.save(model_dir + "/model.keras")

        # Continue training
        print("Continuing training...")
        model.fit(train_gen, validation_data=val_gen, epochs=1)

    # Only keep Original image in validation cache
    val_gen.keep_original()

    zip_model_and_cache(model)


if __name__ == "__main__":
    main()
