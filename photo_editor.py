import random
from PIL import Image

"""
Image representation: A dictionary with "width", "height", and "pixels" as key.
"width" and "height" are integers representing the width and height of the image in pixel.
"pixels" is a list of size "width"*"height", containing the values at each pixel in order 
(left to right, then top to bottom). For greyscale images, those values will be an integer 
in the range [0,255] (0 for black, 255 for white). For color images, it will be a tuple of 
three integers, each in the range [0,255] to represent the red, green, and blue (rgb) values 
respectively of the pixel.

"""

# VARIOUS FILTERS

def get_pixel(image, row, col):
    return image["pixels"][col + image["width"] * row]


def set_pixel(image, row, col, color):
    image["pixels"][col + image["width"] * row] = color


def apply_per_pixel(image, func):
    ''' change pixel in the image'''
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }
    for col in range(image["height"]):
        for row in range(image["width"]):
            color = get_pixel(image, col, row)
            new_color = func(color)
            result["pixels"].append(new_color)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda color: 255 - color)


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", the function
    returns None.

    Otherwise, the output of this function would have the the form of 
    a dictionary with "height", "width", and "pixels" keys, but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers.

    This process would not mutate the input image; rather, it creates a
    separate structure to represent the output.

    """
    if (
        boundary_behavior not in {"zero", "extend","wrap"}
    ):
        return None

    result = {}
    width= image["width"]
    result["width"] = width
    height= image["height"]
    result["height"] = height
    pix = []
    size = int(len(kernel) ** 0.5)

    def correlate_pix(index, pixels, wid):
        """correlate for all pixel not effected by boundary, 
        pixel at index 'index' of image with width wid"""
        color = 0
        ind_k = 0  # index of kernel
        for i in range(int(0.5 * (1 - size)), int(0.5 * (1 + size))):
            ind_p = index + wid * i  # index of pixel
            for j in range(int(0.5 * (1 - size)), int(0.5 * (1 + size))):
                color += pixels[ind_p + j] * kernel[ind_k]
                ind_k += 1
        return color

    def pix_zero():
        """generate extended pixel of zero extension, original 
        with 0.5*(size-1) extension in all direction"""
        pixel = iter(image["pixels"])
        new_pix = []
        new_pix.extend([0] * (int(0.5 * (size - 1)) * (size + width - 1)))
        for _ in range(height):
            new_pix.extend([0] * int(0.5 * (size - 1)))
            for _ in range(width):
                new_pix.append(next(pixel))
            new_pix.extend([0] * int(0.5 * (size - 1)))
        new_pix.extend([0] * (int(0.5 * (size - 1)) * (size + width - 1)))
        return new_pix

    def pix_extend():
        """generate extended pixel for extends extension"""
        new_pix = []
        for i in range(int(0.5 * (size - 1))):
            new_pix.extend([image["pixels"][0]] * int(0.5 * (size - 1)))
            for j in range(width):
                new_pix.append(image["pixels"][j])
            new_pix.extend([image["pixels"][width - 1]] * int(0.5 * (size - 1)))
        for i in range(height):
            new_pix.extend([image["pixels"][i * width]] * int(0.5 * (size - 1)))
            for j in range(width):
                new_pix.append(image["pixels"][j + i * width])
            new_pix.extend([image["pixels"][i * width + width- 1]] * int(0.5 * (size - 1)))
        for i in range(int(0.5 * (size - 1))):
            new_pix.extend([image["pixels"][(height - 1) * width]] * int(0.5 * (size - 1)))
            for j in range(width):
                new_pix.append(image["pixels"][j + (height - 1) * width])
            new_pix.extend([image["pixels"][height * width- 1]] * int(0.5 * (size - 1)))
        return new_pix

    def pix_wrap():
        """generate extended pixel for wrap extension"""
        new_pixel = []
        new_pix = []
        for i in range(height):
            extension = image["pixels"][i * width : i * width + width]
            new_pix.extend(extension[width - int(0.5 * (size - 1)) % width :])
            new_pix.extend(extension * (int((0.5 * (size - 1))) // width))
            for j in range(width):
                new_pix.append(image["pixels"][i * width + j])
            new_pix.extend(extension * (int(0.5 * (size - 1)) // width))
            new_pix.extend(extension[: int(0.5 * (size - 1)) % width])
        length = len(new_pix)
        new_pixel.extend(new_pix[length - (int(0.5 * (size - 1)) % height) * (size + width - 1) :])
        for i in range(int(0.5 * (size - 1)) // height):
            new_pixel.extend(new_pix)
        new_pixel.extend(new_pix)
        for i in range(int(0.5 * (size - 1)) // height):
            new_pixel.extend(new_pix)
        new_pixel.extend(new_pix[: (int(0.5 * (size - 1)) % height) * (size + width - 1)])
        return new_pixel

    if boundary_behavior == "zero":
        pixels = pix_zero()

    elif boundary_behavior == "extend":
        pixels = pix_extend()

    else:
        pixels = pix_wrap()

    for i in range(height):
        for j in range(width):
            pix.append(
                correlate_pix(
                    (int(0.5 * (size - 1)) + i) * (size - 1 + width)
                    + int(0.5 * (size - 1))
                    + j,
                    pixels,
                    size - 1 + width,
                )
            )

    result["pixels"] = pix.copy()
    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure values in the "pixels" list are all
    integers in the range [0, 255].

    All values are converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input would have value
    255 in the output; and any locations with values lower than 0 in the input
    would have value 0 in the output.
    """
    for i in range(len(image["pixels"])):
        if image["pixels"][i] <= 0:
            image["pixels"][i] = 0
        elif image["pixels"][i] > 255:
            image["pixels"][i] = 255
        else:
            image["pixels"][i] = int(round(image["pixels"][i]))


# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process does not mutate the input image; it creates a
    separate structure to represent the output.
    """
    def kernel(size):
        entry = 1 / size**2
        kernel_rep = size**2 * [entry]
        return kernel_rep

    blur = correlate(image, kernel(kernel_size), "extend")
    round_and_clip_image(blur)
    return blur


def sharpened(image, n):
    """take an image, return a sharpend image by kernel of size n"""
    blur = blurred(image, n)
    sharp = {}
    sharp["width"] = image["width"]
    sharp["height"] = image["height"]
    sharp["pixels"] = []
    for i in range(len(image["pixels"])):
        sharp["pixels"].append(2 * image["pixels"][i] - blur["pixels"][i])
    round_and_clip_image(sharp)
    return sharp


def edges(image):
    """return image that emphasize the edge"""
    k1 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    k2 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    o1 = correlate(image, k1, "extend")
    o2 = correlate(image, k2, "extend")
    orc = {}
    orc["height"] = image["height"]
    orc["width"] = image["width"]
    orc["pixels"] = []
    for i in range(len(image["pixels"])):
        orc["pixels"].append((o1["pixels"][i] ** 2 + o2["pixels"][i] ** 2) ** 0.5)
    round_and_clip_image(orc)
    return orc


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def new_filter(image_color):
        pixel = image_color["pixels"]
        new_pixel = []
        new_image = {}
        red, green, blue = [], [], []
        for rd, gr, bl in pixel:
            red.append(rd)
            green.append(gr)
            blue.append(bl)
        image_red, image_green, image_blue = {}, {}, {}
        image_red["height"], image_green["height"], image_blue["height"] = (
            image_color["height"],
            image_color["height"],
            image_color["height"],
        )
        image_red["width"], image_green["width"], image_blue["width"] = (
            image_color["width"],
            image_color["width"],
            image_color["width"],
        )
        image_red["pixels"], image_green["pixels"], image_blue["pixels"] = (
            red,
            green,
            blue,
        )
        new_image["height"], new_image["width"] = (
            image_color["height"],
            image_color["width"],
        )
        new_image_red = filt(image_red)
        new_image_green = filt(image_green)
        new_image_blue = filt(image_blue)
        for i in range(len(pixel)):
            pix = (
                new_image_red["pixels"][i],
                new_image_green["pixels"][i],
                new_image_blue["pixels"][i],
            )
            new_pixel.append(pix)
        new_image["pixels"] = new_pixel
        return new_image

    return new_filter


def make_blur_filter(kernel_size):
    def blurred_2(image):
        return blurred(image, kernel_size)

    return blurred_2


def make_sharpen_filter(kernel_size):
    def sharpened_2(image):
        return sharpened(image, kernel_size)

    return sharpened_2


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def compound_filt(image):
        for filt in filters:
            image = filt(image)
        return image

    return compound_filt


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    for _ in range(ncols):
        grey_im = greyscale_image_from_color_image(image)
        energy = compute_energy(grey_im)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        new_im = image_without_seam(image, seam)
        image = new_im
    return new_im


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    new_image = {"height": image["height"], "width": image["width"], "pixels": []}
    for rd, gr, bl in image["pixels"]:
        new_image["pixels"].append(round(0.299 * rd + 0.587 * gr + 0.114 * bl))
    return new_image


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", using
    the edges function.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map".

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    height, width, pixel = energy["height"], energy["width"], energy["pixels"]
    pix = pixel[:width]
    cumul_energy = {"height": height, "width": width, "pixels": pix}
    for i in range(1, height):
        pix.append(pixel[i * width] + min(pix[(i - 1) * width], pix[(i - 1) * width + 1]))
        for j in range(1, width - 1):
            pix.append(
                pixel[j + i * width]
                + min(
                    pix[j + (i - 1) * width],
                    pix[j + 1 + (i - 1) * width], pix[j - 1 + (i - 1) * width],
                )
            )
        pix.append(pixel[(i + 1) * width - 1] + min(pix[i * width - 1], pix[i * width - 2]))
    return cumul_energy


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam.
    """
    height, width, pixel = cem["height"], cem["width"], cem["pixels"]
    min_btm_ind = width * (height - 1)
    for i in range(width * (height - 1) + 1, width * height):
        if pixel[i] < pixel[min_btm_ind]:
            min_btm_ind = i
    seam = [min_btm_ind]
    for _ in range(1, height):
        if seam[-1] % width == 0:
            min_ind = seam[-1] - width
            if pixel[min_ind + 1] < pixel[min_ind]:
                min_ind += 1
        elif seam[-1] % width < width- 1:
            min_ind = seam[-1] - width - 1
            if pixel[min_ind + 1] < pixel[min_ind]:
                min_ind += 1
            if pixel[seam[-1] - width+ 1] < pixel[min_ind]:
                min_ind = seam[-1] - width + 1
        else:
            min_ind = seam[-1] - width - 1
            if pixel[min_ind + 1] < pixel[min_ind]:
                min_ind += 1
        seam.append(min_ind)
    return seam


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    pixel = image["pixels"].copy()
    new_image = {
        "height": image["height"],
        "width": image["width"] - 1,
        "pixels": pixel,
    }
    for i in seam:
        pixel.pop(i)
    return new_image


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name. If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()