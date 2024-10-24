import cv2
import numpy as np
import os
from typing import Tuple


def get_block_coordinates(block: dict, img_width: int,
                          img_height: int) -> Tuple[float, float,
                                                    float, float]:
    """
    Calculate the coordinates of a block within an image.

    Parameters:
        block (dict): A dictionary representing the block with 'bounds'
                      information.
        img_width (int): The width of the image with the blocks.
        img_height (int): The height of the image with the blocks.

    Returns:
        Tuple[float, float, float, float]: A tuple representing the
                                           coordinates of the block
                                           (xmin, ymin, width, height).
    """

    bounds = block['bounds']
    xmin = int(bounds[0] * img_width)
    ymin = int(bounds[1] * img_height)
    xmax = int(bounds[2] * img_width)
    ymax = int(bounds[3] * img_height)
    w = xmax - xmin
    h = ymax - ymin
    return xmin, ymin, w, h


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
        # cv2.imwrite(f'{url_image_trg}_resized.jpg', img_per)
        return img_per, image_src.shape[1], image_src.shape[0], percent # Returns the original 2-D dimensions

    return image_src, image_src.shape[1], image_src.shape[0], 1 # Image without changes


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


def convert_ordered_block_stack_to_cv2(image: np.ndarray, blocks: list[dict]):
    """
       Convert an ordered block stack to a list of OpenCV images.

       Args:
           image (numpy.ndarray): The original image as a numpy array.
           blocks (list[dict]): The ordered block stack as a list of dictionaries, where each dictionary represents a block and contains keys "x", "y", "w", and "h".

       Returns:
           list[numpy.ndarray]: A list of OpenCV images, where each image corresponds to a block in the ordered block stack.
    """

    img_height, img_width, _ = image.shape

    cut_blocks = []
    max_width = 0
    cut_blocks_cv2 = []

    for block in blocks:
        x1, y1, w, h = get_block_coordinates(block, img_width, img_height)

        # Calculate the bottom-right corner coordinates of each block
        x2 = x1 + w
        y2 = y1 + h

        # Cut the text block region from the image
        cut_block = image[y1:y2, x1:x2]

        # Track the maximum width of the blocks
        if cut_block.shape[1] > max_width:
            max_width = cut_block.shape[1]

        # Append the cut block to the list and transform into image cv2
        cut_blocks.append(cut_block)
        image_block = cv2.cvtColor(cut_block, cv2.COLOR_RGB2BGR)
        image_cv2 = cv2.imdecode(image_block.astype(np.uint8), cv2.IMREAD_COLOR)
        cut_blocks_cv2.append(image_cv2)

    return cut_blocks_cv2
