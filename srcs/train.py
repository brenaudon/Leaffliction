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
    - utils (custom module for utility functions)
    - generator (custom module for data generation)
    - save_zip (custom module for saving model and zip it with cache)
"""

import os
import random
import argparse
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from input_pipeline import make_dataset
from save_zip import zip_model_and_cache
from preprocess_dataset import build_prepared_dataset


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


def get_train_val_split(data_dir, class_names, cache_dir,
                        force_new_split: bool):
    """
    Get training and validation samples from the data directory.
    This function gathers all samples from the data directory,
    and if `force_new_split` is True, it creates a new train/validation split.
    If `force_new_split` is False, it attempts to load existing split files
    from a cache directory.

    @param data_dir: Directory containing class sub-folders
    @type data_dir: str
    @param class_names: List of class names (sub-folder names)
    @type class_names: list
    @param cache_dir: Directory to save or load the train/val split files
    @type cache_dir: str
    @param force_new_split: If True, create a new train/val split
                            even if existing files are found
    @type force_new_split: bool

    @return: Tuple of (train_samples, val_samples)
    @rtype: tuple
    """
    train_file = os.path.join(cache_dir, "train_split.txt")
    val_file = os.path.join(cache_dir, "val_split.txt")

    # use existing txt files
    if (not force_new_split and
            os.path.exists(train_file) and os.path.exists(val_file)):
        with open(train_file) as f:
            train_paths = [ln.strip() for ln in f if ln.strip()]
        with open(val_file) as f:
            val_paths = [ln.strip() for ln in f if ln.strip()]

        train_samples = [(p, os.path.basename(os.path.dirname(p)))
                         for p in train_paths]
        val_samples = [(p, os.path.basename(os.path.dirname(p)))
                       for p in val_paths]

        print(f"Loaded split from {train_file} / {val_file} "
              f"({len(train_samples)} train, {len(val_samples)} val)")
        return train_samples, val_samples

    # build a fresh split
    all_samples = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_samples.append((os.path.join(cls_dir, fn), cls))

    random.seed(42)
    random.shuffle(all_samples)
    cut = int(0.9 * len(all_samples))
    train_samples = all_samples[:cut]
    val_samples = all_samples[cut:]

    # save lists to txt (absolute paths)
    with open(train_file, "w") as f:
        f.writelines(p[0] + "\n" for p in train_samples)
    with open(val_file, "w") as f:
        f.writelines(p[0] + "\n" for p in val_samples)

    print(f"Created new split → {train_file} / {val_file} "
          f"({len(train_samples)} train, {len(val_samples)} val)")
    return train_samples, val_samples


def get_args():
    """
    Parse command line arguments for training the CNN model.

    @return: Parsed arguments
    @rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Train or resume training"
                                             " of the CNN.")
    ap.add_argument("-i", "--input-dir", required=True,
                    help="input directory containing one sub-folder per class")
    ap.add_argument("-m", "--model",
                    help="existing .keras file to resume training from")
    ap.add_argument("-e", "--epochs", type=int, default=1,
                    help="number of epochs to train the model before asking"
                         " user input to continue or stop (default: 1)")
    return ap.parse_args()


def main():
    """
    Main function to train the model.
    """
    args = get_args()
    data_dir = args.input_dir
    nb_epochs = args.epochs
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    model_dir = "model"
    cache_dir = "cache"
    create_split = True if not args.model else False

    num_classes = len(os.listdir(data_dir))
    class_names = sorted(os.listdir(data_dir))

    # save class names to a file (create if not exists)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    # Build prepared dataset if not exists (transformed images)
    prepared_dataset_dir = "cache/prepared"
    if not os.path.exists(os.path.join(prepared_dataset_dir, "manifest.json")):
        print("cache/prepared/ not found: running preprocessing")
        build_prepared_dataset(input_dir="images",
                               output_dir=prepared_dataset_dir)

    # Get training and validation samples
    train_samples, val_samples = get_train_val_split(
        data_dir, class_names, cache_dir, create_split)

    train_ds = make_dataset(train_samples, shuffle=True)
    val_ds = make_dataset(val_samples, shuffle=False)

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

    model.fit(train_ds, validation_data=val_ds, epochs=nb_epochs)

    epoch = nb_epochs

    while True:
        user_input = input(f"Epoch {epoch + 1}."
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
        model.fit(train_ds, validation_data=val_ds, epochs=1)

        epoch += 1

    print("Zipping model and cache...")
    zip_model_and_cache()


if __name__ == "__main__":
    main()
