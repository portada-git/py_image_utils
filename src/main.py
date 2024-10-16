from utils.image_utilities_cv import resize_image_percent, reduce_image_resolution, rescale_mask
import numpy as np

IMG_SRC = 'C:\\Users\\VLIR-Toledano\\Desktop\\dataimg\\1850_01_01_HAB_DM_00000_U_1_0.jpg'
IMG_DEST = 'C:\\Users\\VLIR-Toledano\\Desktop\\dataimg\\img1'


def simulate_layout_mask(height, width, num_blocks=5):
    # Create an empty mask
    layout_mask = np.zeros((height, width), dtype=np.int32)

    # Randomly generate block regions
    for block_id in range(1, num_blocks + 1):
        # Generate random coordinates for block boundaries
        xmin = np.random.randint(0, width // 2)
        ymin = np.random.randint(0, height // 2)
        xmax = np.random.randint(width // 2, width)
        ymax = np.random.randint(height // 2, height)

        # Assign block_id to the selected region in the mask
        layout_mask[ymin:ymax, xmin:xmax] = block_id

    return layout_mask


# testing methods
if __name__ == "__main__":
    img_low, original_width, original_height = resize_image_percent(IMG_SRC, IMG_DEST, percent=0.20)

    # makes inference via arcanum, in this we simulate layout mask
    mask_simulated = simulate_layout_mask(img_low.shape[1], img_low.shape[0])

    rescaled_mask = rescale_mask(mask_simulated, original_width, original_height)

    print("Shape of scaled mask: ", rescaled_mask.shape)
    print("Shape of original image: ", (original_height, original_width))
    print("Shape of resized image: ", img_low.shape[:2])