"""
This script prepares a dataset of leaf images by applying various
transformations and saving the results in a structured format.
Can also be called from the command line to process images in a directory
and save the prepared dataset in a specified output directory.

Dependencies:
    - cv2
    - json
    - argparse
    - numpy
    - tqdm
    - pathlib
    - Transformation (custom module for image transformations)
"""

import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from Transformation import apply_transformations


IMG_KEYS = ["Original", "Gaussian Blur", "Mask",
            "ROI", "Analyze Objects", "Landmark"]
HIST_NAME = "hist"


def build_prepared_dataset(input_dir, output_dir="cache/prepared"):
    """
    Create prepared/<class>/<name>_{key}.npy and manifest.json
    files from the input directory containing leaf images.
    The input directory should contain subdirectories for each class,
    where each subdirectory contains images of that class.

    @param input_dir: Path to the input directory
    @type input_dir: str
    @param output_dir: Path to the output directory
    @type output_dir: str

    @return: None
    """
    src_root = Path(input_dir)
    dst_root = Path(output_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)

    classes = sorted([d.name for d in src_root.iterdir() if d.is_dir()])
    class_idx = {c: i for i, c in enumerate(classes)}
    manifest = []

    print(f"Building prepared dataset in {dst_root}")
    for cls in classes:
        files = [p for p in (src_root / cls).rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print(f"{cls:20s} ({len(files)} images)")
        for img_path in tqdm(files, unit="img", leave=False):
            name = img_path.stem
            rel = Path(cls) / name

            trans = apply_transformations(str(img_path), plot=False)

            out_dir = dst_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            # 6 images
            for key in IMG_KEYS:
                if key not in trans:
                    continue
                path = out_dir / f"{name}_{key}.jpg"
                cv2.imwrite(str(path), trans[key])

            # histogram
            hist_vec = (trans["Color_Histogram"].iloc[:, 1]
                        .values.astype("float32"))
            np.save(out_dir / f"{name}_{HIST_NAME}.npy", hist_vec)

            manifest.append({"rel": str(rel), "label": class_idx[cls]})

    with open(dst_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Prepared {len(manifest)} samples.\n")


def _parse_args():
    """
    Parse command line arguments for the dataset preparation script.

    @return: Parsed arguments
    @rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", required=True)
    p.add_argument("-o", "--output-dir", default="cache/prepared")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    build_prepared_dataset(a.input_dir, a.output_dir)
