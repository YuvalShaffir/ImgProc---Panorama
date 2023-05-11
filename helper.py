from scipy.signal import convolve2d
import numpy as np
from skimage.color import rgb2gray
import imageio
from scipy.signal import convolve2d
from scipy import signal
from scipy.ndimage.filters import convolve

LOWEST_SHAPE_SIZE = 16
GRAYSCALE = 1
RGB = 2
DEFAULT_KERNAL_ARRAY = np.array([1, 1])
DEFAULT_KERNAL_LENGTH = 2


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # read the image - check if it is greyscale image:
    img = imageio.imread(filename)

    # greyscale representation
    if representation == GRAYSCALE:
        img_g = rgb2gray(img)
        img_g = img_g.astype('float64')
        return img_g

    # RGB representation
    if representation == RGB:
        img_rgb = img.astype('float64')
        img_rgb_norm = img_rgb / 255
        return img_rgb_norm


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurred_image = convolve(convolve(im, blur_filter, mode='constant'), blur_filter.T, mode='constant')
    # plt.imshow(blurred_image, cmap='gray')
    # plt.show()
    return blurred_image[::2, ::2]


def blur_filter_generator(filter_size):
    """generates a blur filter vector from filter size"""
    kernel = DEFAULT_KERNAL_ARRAY
    vec_length = DEFAULT_KERNAL_LENGTH
    filter_vec = kernel

    while vec_length < filter_size:
        filter_vec = signal.convolve(filter_vec, kernel)
        vec_length += 1

    filter_vec = filter_vec / sum(filter_vec)  # normalization

    return filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = []
    filter_vec = blur_filter_generator(filter_size).reshape((1, filter_size))
    count = 0

    while count < max_levels:
        if im.shape[0] <= LOWEST_SHAPE_SIZE or im.shape[1] <= LOWEST_SHAPE_SIZE:
            break
        pyr.append(im)
        im = reduce(im, filter_vec)
        count += 1

    return pyr, filter_vec


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

