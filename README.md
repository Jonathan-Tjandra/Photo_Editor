# Python Pixel-Based Photo Editor

A lightweight, dependency-free photo editor written in Python. This project manipulates images by representing them as a dictionary of pixel values, applying various mathematical transformations, and then reconstructing the image. It uses the Python Imaging Library (PIL) or its modern fork, Pillow, for loading and saving image files.

## Features

This editor supports a variety of features for both grayscale and color images:

* **Color Inversion:** Reverses the colors in an image.
* **Brightness Adjustment:** Modifies the overall brightness (implemented via correlation).
* **Grayscale Conversion:** Converts a color image into a grayscale image.
* **Edge Emphasis:** Applies a Sobel operator to highlight the edges within an image.
* **Blur & Sharpen:** Applies blur and sharpen filters using kernel correlation.
* **Cropping (Seam Carving):** Implements "smart cropping" by using the seam carving algorithm to remove the least important columns from an image.

## How It Works

The core of this project is a custom image representation: a Python dictionary with three keys:
-   `"width"`: The width of the image in pixels.
-   `"height"`: The height of the image in pixels.
-   `"pixels"`: A flat list of pixel values. For grayscale images, each value is an integer `[0-255]`. For color images, each value is a tuple of three integers `(R, G, B)`.

All filtering and manipulation functions operate on this dictionary structure, providing a clear and understandable way to process image data at the pixel level.

## Installation

To use this photo editor, you need to have Python 3 and the Pillow library installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jonathan-Tjandra/Photo_Editor.git
    cd Photo_Editor
    ```

2.  **Install the required package:**
    ```bash
    pip install -r requirements.txt
    ```

## Basic Usage

The provided Python script contains several functions that can be imported and used to manipulate images. The main workflow is:

1.  Load an image using `load_color_image()` or `load_greyscale_image()`.
2.  Apply one or more filter functions (e.g., `inverted()`, `edges()`, `blurred()`).
3.  Save the resulting image dictionary using `save_color_image()` or `save_greyscale_image()`.

```python
from photo_editor import load_color_image, save_color_image, color_filter_from_greyscale_filter, edges

# Load an image
original_image = load_color_image('my_photo.png')

# Create a color-compatible edge detection filter
color_edges_filter = color_filter_from_greyscale_filter(edges)

# Apply the filter
edited_image = color_edges_filter(original_image)

# Save the new image
save_color_image(edited_image, 'my_photo_edges.png')

print("Successfully applied edge detection filter and saved the image.")
```

## Acknowledgements

This project uses the **Pillow** library, a fork of the Python Imaging Library (PIL), for handling image file I/O.