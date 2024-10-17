import cv2
import numpy as np
import os

def resize_image_percent(url_image_src, url_image_trg, percent=0.75):
    """
    Resizes an image by a given percentual ratio.

    Args:
        url_image_src (str): The URL of the input image file.
        url_image_trg (str): The URL of the output image file.
        percent (float, optional): The resizing ratio. Defaults to 0.75.

    Returns:
        img_per (numpy.ndarray): The resized image as a numpy array.
        original_width (int): The width of the original image.
        original_height (int): The height of the original image
    """
    img = cv2.imread(url_image_src)
    img_per = cv2.resize(img, None, fx=percent, fy=percent)
    cv2.imwrite(f'{url_image_trg}_resized.jpg', img_per)

    return img_per, img.shape[1], img.shape[0]


def resize_image_percent_til_size(image_src, url_image_trg, nested_size=6291456): # 6MB
    """
    Resizes an image by a percentual ratio until reaches max_size.

    Args:
        image_src (str): Input image object.
        url_image_trg (str): The URL of the output image file.
        nested_size (int, optional): Size in bytes

    Returns:
        img_per (numpy.ndarray): The resized image as a numpy array.
        original_width (int): The width of the original image.
        original_height (int): The height of the original image
    """

    size_in_bytes = cv2.imencode(".jpg", image_src)[1].size
    percent = nested_size / size_in_bytes
    if size_in_bytes > nested_size:
        img_per = cv2.resize(image_src, None, fx=percent, fy=percent)
        cv2.imwrite(f'{url_image_trg}_resized.jpg', img_per)
        return img_per, img.shape[1], img.shape[0], percent # Returns the original 2-D dimensions

    return image_src, img.shape[1], img.shape[0], 1 # Image without changes


def reduce_image_resolution(url_image_src, url_image_trg, quality=30):
    """
    Reduce the resolution of an image by saving it with a lower quality JPEG compression.

    Args:
        url_image_src (str): The URL of the input image file.
        url_image_trg (str): The URL of the output image file.
        quality (int, optional): The JPEG compression quality (1-100). Defaults to 30.

    Returns:
        img_low (numpy.ndarray): The reduced resolution image as a numpy array.
        quality (int): The JPEG compression quality used for the output file.
        original_width (int): The width of the original image.
        original_height (int): The height of the original image
    """
    img = cv2.imread(url_image_src)
    cv2.imwrite(f'{url_image_trg}_low_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img_low = cv2.imread(f'{url_image_trg}_low_quality.jpg')

    return img_low, img.shape[1], img.shape[0]


def rescale_mask(layout_mask_arcanum, original_width, original_height):
    """
        Rescale a layout mask to match the original image size.

        Args:
            layout_mask_arcanum (numpy.ndarray): The layout mask as a numpy array.
            original_width (int): The width of the original image.
            original_height (int): The height of the original image.

        Returns:
            layout_mask (numpy.ndarray): The rescaled layout mask as a numpy array.
    """
    layout_mask = cv2.resize(layout_mask_arcanum, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    return layout_mask
