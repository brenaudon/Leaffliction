"""
This script defines a function to save a Keras model and its cache into a zip.
Also generates SHA-1 signature for the zip file and saves it to signature.txt.

Dependencies:
    - shutil
    - hashlib
"""

import shutil
import hashlib


def zip_model_and_cache(
        model_dir="model",
        cache_dir="cache",
        zip_name="leaffliction_package.zip"):
    """
    Save the model and cache to a zip file. Creates signature.txt with SHA-1

    @param model: The trained model to save.
    @type model: keras.Model
    @param model_dir: Directory to save the model.
    @type model_dir: str
    @param cache_dir: Directory containing the cache to include in the zip.
    @type cache_dir: str
    @param zip_name: Name of the output zip file.
    @type zip_name: str

    @return: None
    """

    # Zip model and cache
    print(f"Creating ZIP file: {zip_name}")
    with shutil.make_archive("leaffliction_package",
                             'zip',
                             root_dir='.',
                             base_dir=model_dir) as _:
        pass
    with shutil.make_archive("cache_backup",
                             'zip',
                             root_dir='.',
                             base_dir=cache_dir) as _:
        pass

    # Merge both zips into one
    with open("leaffliction_package.zip", "wb") as out_zip:
        for name in ["leaffliction_package.zip", "cache_backup.zip"]:
            with open(name, "rb") as part:
                shutil.copyfileobj(part, out_zip)

    print("ZIP package created: leaffliction_package.zip")

    # Optionally, generate SHA-1 signature
    sha1 = hashlib.sha1()
    with open("leaffliction_package.zip", "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    signature = sha1.hexdigest()

    with open("signature.txt", "w") as sig_file:
        sig_file.write(signature + "\n")

    print(f"SHA-1 Signature: {signature} (written to signature.txt)")
