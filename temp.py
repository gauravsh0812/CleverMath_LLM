import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

IMAGE = Image.open(f"data/images/1086.png")
IMAGE.show()

# checking the size of the image
w, h = IMAGE.size

# Desired image size
desired_width = 480
desired_height = 320

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
new_image = Image.new('RGBA', (desired_width, desired_height), (255, 255, 255, 255)) # White background

# Paste the original image onto the new image, centered and padded as needed
new_image.paste(IMAGE, (0, padding_top))

# Save the new image
new_image.show()

# convert to tensor
convert = transforms.ToTensor()
IMAGE = convert(IMAGE)
print(IMAGE.shape)

NIMAGE = convert(new_image)
print(NIMAGE.shape)


# transform = T.ToPILImage()
# img = transform(tnsr)
# img.show()