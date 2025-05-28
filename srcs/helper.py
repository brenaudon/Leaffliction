"""
This script deletes augmented images from a specified directory.

Dependencies:
    - os
    - sys
    - Augmentation (custom module)
"""

import os
import sys
from Augmentation import is_directory


def delete_augmented_images(directory):
    """
    Delete all image files in the directory (and subdirectories)
    that have an underscore in their filename (e.g., *_Flip.jpg).

    @param directory: Root path to scan
    @type directory: str
    """
    image_extensions = ('.jpg', '.jpeg', '.png')

    deleted_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if '_' in file and file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"[⚠️] Failed to delete {file_path}: {e}")

    print(f"\n✅ Finished. {deleted_count} files deleted.")


def main():
    """
    Main function to execute the script.

    @return: None
    """
    if len(sys.argv) != 2:
        print("Usage: python helper.py "
              "<path_to_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if is_directory(input_path):
        delete_augmented_images(input_path)
    else:
        print(f"[⚠️] {input_path} is not a directory.")


if __name__ == "__main__":
    main()
