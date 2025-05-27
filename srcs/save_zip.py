"""
This script defines a function to put a Keras model and its cache into a zip.
Also generates SHA-1 signature for the zip file and saves it to signature.txt.

Dependencies:
    - shutil
    - hashlib
    - zipfile
"""

from pathlib import Path
import hashlib
import zipfile


def add_folder_to_zip(zip_obj, folder_path: Path, base_folder_name: str):
    """
    Recursively add a folder to an open ZipFile, preserving its name

    @param zip_obj: Open zipfile.ZipFile object
    @type zip_obj: zipfile.ZipFile
    @param folder_path: Path to the folder to add
    @type folder_path: Path
    @param base_folder_name: Base name to use in the archive
    """
    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            # Build archive name:  model/relative/path/in/folder
            name = Path(base_folder_name) / file_path.relative_to(folder_path)
            zip_obj.write(file_path, name)


def zip_model_and_cache(
        model_dir="model",
        cache_dir="cache",
        zip_name="leaffliction_package.zip"):
    """
    Put the model and cache to a zip file. Creates signature.txt with SHA-1

    @param model_dir: Directory containing the model.
    @type model_dir: str
    @param cache_dir: Directory containing the cache.
    @type cache_dir: str
    @param zip_name: Name of the output zip file.
    @type zip_name: str

    @return: None
    """

    model_dir = Path(model_dir)
    cache_dir = Path(cache_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    print(f"Creating ZIP file: {zip_name}")

    with zipfile.ZipFile(zip_name, "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        print("Adding model directory to ZIP...")
        add_folder_to_zip(zf, model_dir, "model")
        print("Adding cache directory to ZIP...")
        add_folder_to_zip(zf, cache_dir, "cache")

    print(f"ZIP package created: {zip_name}")

    # Optionally, generate SHA-1 signature
    sha1 = hashlib.sha1()
    with open(zip_name, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    signature = sha1.hexdigest()

    with open("signature.txt", "w") as sig_file:
        sig_file.write(signature + "\n")

    print(f"SHA-1 Signature: {signature} (written to signature.txt)")


def main():
    zip_model_and_cache()


if __name__ == "__main__":
    main()
