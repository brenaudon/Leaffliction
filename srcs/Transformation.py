import os
import cv2
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from scipy.signal import savgol_filter


def analyze_image(image, mask):
    """
    Analyze the image and mask to find contours, draw lines, and compute PCA.

    @param image: Input image
    @type image: numpy.ndarray
    @param mask: Binary mask for the image
    @type mask: numpy.ndarray

    @return: Image with contours and PCA lines drawn
    @rtype: numpy.ndarray
    """
    # Find contours using OpenCV
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    # Draw contours
    cv2.drawContours(output, contours, -1, (0, 0, 255), thickness=3)

    # Get largest contour (leaf)
    cnt = max(contours, key=cv2.contourArea)

    # get the bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    pt_h_1 = (x, y)
    pt_h_2 = (x + w, y)
    cv2.line(output, pt_h_1, pt_h_2, (255, 0, 255), 3)

    # Compute horizontal center (exact middle of image)
    image_center_x = image.shape[1] // 2

    pt_v_1 = (image_center_x, y)
    pt_v_2 = (image_center_x, y + h)

    # Draw vertical line
    cv2.line(output, pt_v_1, pt_v_2, (255, 0, 255), 3)

    center_x = x + w // 2
    center_y = y + h // 2

    # Draw centroid
    centroid = (int(center_x), int(center_y))
    cv2.circle(output, centroid, 10, (255, 0, 255), -1)

    # draw convex hull instead of raw contour
    hull = cv2.convexHull(cnt)
    cv2.drawContours(output, [hull], -1, (255, 0, 255), 3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    leaf_contour = max(contours, key=cv2.contourArea)

    # Convert contour points to 2D array for PCA
    data_pts = leaf_contour[:, 0, :].astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))

    # Draw the main axis (midrib approximation)
    center = tuple(map(int, mean[0]))
    p1 = (
        int(center[0] + 200 * eigenvectors[0, 0]),
        int(center[1] + 200 * eigenvectors[0, 1])
    )
    p2 = (
        int(center[0] - 200 * eigenvectors[0, 0]),
        int(center[1] - 200 * eigenvectors[0, 1])
    )

    cv2.line(output, p1, p2, (255, 0, 255), 3)

    return output


def get_color_histogram(img):
    """
    Compute the color histogram of an image in different color spaces.

    @param img: Input image
    @type img: numpy.ndarray

    @return: DataFrame containing histogram data
    @rtype: pandas.DataFrame
    """
    # Convert the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Prepare channels from different color spaces
    channels = {
        "blue": img_rgb[:, :, 2],
        "red": img_rgb[:, :, 0],
        "green": img_rgb[:, :, 1],
        "hue": img_hsv[:, :, 0],
        "saturation": img_hsv[:, :, 1],
        "value": img_hsv[:, :, 2],
        "lightness": img_hls[:, :, 1],
        "green-magenta": img_lab[:, :, 1],
        "blue-yellow": img_lab[:, :, 2],
    }

    # Histogram computation function
    def get_histogram(channel):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum() * 100  # Percentage
        hist = savgol_filter(hist, window_length=11, polyorder=4)
        return hist

    # Collect histogram data
    hist_data = []
    for name, channel in channels.items():
        hist = get_histogram(channel)
        for intensity, value in enumerate(hist):
            hist_data.append({
                "Pixel intensity": intensity,
                "Proportion of pixels (%)": value,
                "color Channel": name
            })

    df_hist = pd.DataFrame(hist_data)
    return df_hist


def plot_color_histogram(df_hist, mode='plot', save_path=None):
    """
    Plot the color histogram data.

    @param df_hist: DataFrame containing histogram data
    @type df_hist: pandas.DataFrame

    @return: None
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_hist, x="Pixel intensity",
                 y="Proportion of pixels (%)", hue="color Channel")
    plt.title("Color Histogram")
    plt.ylim(0, 10)
    plt.grid(True)
    plt.tight_layout()
    if mode == 'save':
        plt.savefig(save_path)
        plt.close()
    elif mode == 'plot':
        plt.show()


def save_transformed_data(tranform_result, image_name, save_path):
    """
    Save the transformed images to the specified directory.

    @param tranform_result: Dictionary of transformed images
    @type tranform_result: dict
    @param image_name: Name of the original image
    @type image_name: str
    @param save_path: Directory to save the transformed images
    @type save_path: str

    @return: None
    """
    os.makedirs(save_path, exist_ok=True)
    for key, result_img in tranform_result.items():
        save_path_img = os.path.join(save_path, f"{image_name}_{key}.png")
        if key == 'Original':
            pass
        elif key == 'Color_Histogram':
            plot_color_histogram(tranform_result['Color_Histogram'],
                                 mode='save',
                                 save_path=save_path_img)
        else:
            cv2.imwrite(save_path_img, result_img)


def plot_results(results, image_name):
    """
    Plot the results of the transformations.

    @param results: Dictionary of transformed images
    @type results: dict
    @param image_name: Name of the original image
    @type image_name: str

    @return: None
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Transformations for: {image_name}", fontsize=16)

    axs = axs.flatten()
    for ax, (key, img) in zip(axs, results.items()):
        if key == 'Color_Histogram':
            pass
        else:
            ax.set_title(key)
            ax.imshow(img,
                      cmap='gray' if len(img.shape) == 2 else None)
            ax.axis('off')

    # hide the empty subplots
    for i in range(len(results) - 1, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    plot_color_histogram(results['Color_Histogram'])


def apply_transformations(
        image_path,
        save_results=False,
        dst_directory=None,
        tags=None
):
    """
    Apply transformations to image and save or display the results if needed.

    @param image_path: Path to the input image
    @type image_path: str
    @param save_results: Flag to save results or not
    @type save_results: bool
    @param dst_directory: Directory to save results
    @type dst_directory: str
    @param tags: Dictionary of transformation flags
    @type tags: dict

    @return: None
    """
    image = cv2.imread(image_path)
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)

    # Create output paths
    results = {'Original': image}

    # Create a mask
    s = pcv.rgb2gray_hsv(image, 's')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=65,
                                    object_type='light')
    if tags is None or tags['gaussian']:
        gaussian_mask = pcv.gaussian_blur(img=s_thresh, ksize=(3, 3),
                                          sigma_x=0, sigma_y=None)
        results['Gaussian Blur'] = gaussian_mask

    blur_mask = pcv.median_blur(s_thresh, ksize=(5, 5))

    # Masked leaf
    if tags is None or tags['mask']:
        masked = image.copy()
        masked[blur_mask < 100] = [255, 255, 255]
        results['Mask'] = masked

    if tags is None or tags['roi'] or tags['analyze']:
        img_lab = pcv.rgb2gray_lab(rgb_img=image, channel='a')
        s_thresh = pcv.threshold.binary(gray_img=img_lab,
                                        threshold=120, object_type='dark')
        mask = pcv.gaussian_blur(img=s_thresh, ksize=(3, 3),
                                 sigma_x=0, sigma_y=None)
        better_mask = pcv.apply_mask(blur_mask, mask, 'black')

        # Region of Interest (ROI) extraction
        if tags is None or tags['roi']:
            roi_img = image.copy()
            roi_img[better_mask > 100] = [0, 255, 0]
            results['ROI'] = roi_img

        # Analyze Objects
        if tags is None or tags['analyze']:
            img_analyzed = analyze_image(image, better_mask)
            results['Analyze Objects'] = img_analyzed

    # Pseudolandmarks
    if tags is None or tags['landmark']:
        top, bottom, center_v = (
            pcv.homology.x_axis_pseudolandmarks(image, blur_mask))
        image_copy = image.copy()
        for i in range(len(top)):
            cv2.circle(image_copy, (int(top[i][0][0]),
                                    int(top[i][0][1])),
                       3, (0, 0, 255), -1)
        for i in range(len(bottom)):
            cv2.circle(image_copy, (int(bottom[i][0][0]),
                                    int(bottom[i][0][1])),
                       3, (255, 0, 255), -1)
        for i in range(len(center_v)):
            cv2.circle(image_copy, (int(center_v[i][0][0]),
                                    int(center_v[i][0][1])),
                       3, (255, 128, 0), -1)
        results['Landmark'] = image_copy

    # Color Histogram
    if tags is None or tags['hist']:
        color_hist = get_color_histogram(image)
        results['Color_Histogram'] = color_hist

    if save_results and dst_directory:
        save_transformed_data(results, name, dst_directory)
    else:
        plot_results(results, name)


def process_directory(src, dst, tags):
    """
    Process all images in a directory and apply transformations.

    @param src: Source directory containing images
    @type src: str
    @param dst: Destination directory to save processed results
    @type dst: str
    @param tags: Dictionary of transformation flags
    @type tags: dict

    @return: None
    """
    for root, _, files in os.walk(src):
        print(f"Processing directory: {root}")
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, src)
                dst_image_dir = os.path.join(dst, relative_path)
                apply_transformations(src_image_path,
                                      save_results=True,
                                      dst_directory=dst_image_dir,
                                      tags=tags)


def print_usage():
    """
    Print the usage instructions for the script.

    @return: None
    """
    print("Usage:")
    print("  python Transformation.py -src <source_directory>"
          " -dst <destination_directory> [tags]")
    print("  python Transformation.py <single_image_path>")


def main():
    """
    Main function to execute the script.

    @return: None
    """
    parser = argparse.ArgumentParser(
        description="Apply transformations "
                    "to a single image or a directory of images."
    )
    parser.add_argument('-src', '--source',
                        help='Source directory of images')
    parser.add_argument('-dst', '--destination',
                        help='Destination directory to save processed results')

    # Individual transformation flags
    parser.add_argument('-all', action='store_true',
                        help='Apply all transformation')
    parser.add_argument('-gaussian', action='store_true',
                        help='Apply Gaussian blur transformation')
    parser.add_argument('-mask', action='store_true',
                        help='Apply mask transformation')
    parser.add_argument('-analyze', action='store_true',
                        help='Apply object analysis')
    parser.add_argument('-roi', action='store_true',
                        help='Apply region of interest extraction')
    parser.add_argument('-landmark', action='store_true',
                        help='Apply landmark detection')
    parser.add_argument('-hist', action='store_true',
                        help='Compute color histogram')

    parser.add_argument('image', nargs='?',
                        help='Path to a single image file')

    args = parser.parse_args()

    tags = {
        'gaussian': args.gaussian or args.all,
        'mask': args.mask or args.all,
        'analyze': args.analyze or args.all,
        'roi': args.roi or args.all,
        'landmark': args.landmark or args.all,
        'hist': args.hist or args.all
    }

    if args.source and args.destination:
        process_directory(args.source, args.destination, tags)
    elif args.image and not args.source and not args.destination:
        apply_transformations(args.image, save_results=False)
    else:
        print_usage()
        parser.print_help()


if __name__ == "__main__":
    main()
