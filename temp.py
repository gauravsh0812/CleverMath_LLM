import os
import torch
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool

# _path = "/groups/claytonm/gauravs_data/clevrmath_data/"
_path = "/home/gauravs/data/clevrmath_data"

def getting_image_tensors():
    """
    we don't need to crop the image but we will 
    pad them to be of the same size i.e. w=480, h=320.
    And it has a lot of wide spaces near the edges.
    """

    print("creating image tensors...")

    images = os.listdir(f"{_path}/images")

    # create an image_tensors folder
    if not os.path.exists(f"{_path}/image_tensors"):
        os.mkdir(f"{_path}/image_tensors")

    with Pool(20) as pool:
        result = pool.map(preprocess_images, images) 

def preprocess_images(img):
    
    IMAGE = Image.open(f"{_path}/images/{img}")
    
    # checking the size of the image
    w, h = IMAGE.size

    # Desired image size
    desired_width = 480
    desired_height = 320

    if w != 480 or h != 320:

        # Original image size
        original_width, original_height = IMAGE.size
        
        # Calculate aspect ratio of the original image
        aspect_ratio = original_width / original_height

        # Calculate the new dimensions while maintaining the aspect ratio
        new_width = desired_width
        new_height = int(new_width / aspect_ratio)

        # Calculate the padding needed to achieve the desired dimensions
        padding_height = desired_height - new_height
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top

        # Create a new image with the desired dimensions
        # the images are 4D - RGB and Alpha (could be transparency)
        new_image = Image.new('RGBA', (desired_width, desired_height), (255, 255, 255, 255)) # White background

        # Paste the original image onto the new image, centered and padded as needed
        new_image.paste(IMAGE, (0, padding_top))

        # Save the new image
        new_image.save(f'{_path}/padded_images/{img}')

        IMAGE = new_image

        # Create an attention mask image
        mask_image = Image.new('L', (desired_width, desired_height), 0) # 'L' mode for grayscale

        # Fill the original content area with 1
        mask_image.paste(1, (0, padding_top, desired_width, new_height + padding_top))

        # Save the mask image
        mask_image.save(f'{_path}/attention_masks/{img}')


    # convert to tensor
    convert = transforms.ToTensor()
    IMAGE = convert(IMAGE)

    # saving the image 
    torch.save(IMAGE, f"{_path}/image_tensors/{img.split('.')[0]}.pt")

getting_image_tensors()