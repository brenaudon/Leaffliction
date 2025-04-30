"""
This script counts the number of images in each class directory
and plots the distribution of images in a pie chart and a bar chart.


Dependencies:
    - os
    - sys
    - matplotlib.pyplot
"""

import os
import sys
import matplotlib.pyplot as plt


def count_images(base_dir):
    """
    Count the number of images in each class directory.

    @param base_dir: Path to the base directory containing class subdirectories
    @type base_dir: str

    @return: Dictionary with class names as keys and image counts as values
    @rtype: dict
    """
    data = {}

    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue  # skip the top-level directory
        class_name = os.path.basename(root)
        image_count = sum(1 for f in files if f.lower()
                          .endswith(('.jpg', '.jpeg', '.png')))
        data[class_name] = data.get(class_name, 0) + image_count

    return data


def plot_distribution(data, plant_name):
    """
    Plot the distribution of images in a pie chart and a bar chart.

    @param data: Dictionary with class names as keys and image counts as values
    @type data: dict
    @param plant_name: Name of the plant for the title
    @type plant_name: str

    @return: None
    """
    labels = list(data.keys())
    counts = list(data.values())

    # Generate consistent colors
    colormap = plt.colormaps['prism']
    colors = [colormap(i / len(labels)) for i in range(len(labels))]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{plant_name} Class Distribution", fontsize=16)

    # Pie chart
    axs[0].pie(counts, labels=labels, autopct='%1.1f%%',
               startangle=140, colors=colors)
    axs[0].axis('equal')  # Equal aspect ratio for pie

    # Bar chart
    bars = axs[1].bar(labels, counts, color=colors)
    axs[1].set_ylabel("Number of Images")
    axs[1].tick_params(axis='x', rotation=45)

    # Add text labels above each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axs[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()


def main():
    """
    Main function to execute the script.

    @return: None
    """
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <path_to_directory>")
        return

    base_dir = sys.argv[1]

    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        return
    plant_name = os.path.basename(base_dir.rstrip('/'))

    data = count_images(base_dir)
    if not data:
        print("No image data found.")
        return

    plot_distribution(data, plant_name)


if __name__ == "__main__":
    main()
