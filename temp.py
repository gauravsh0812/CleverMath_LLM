import torch, torchvision
import torchvision.transforms as T
from PIL import Image

transform = T.ToPILImage()
tnsr = torch.load("segmented_images/tnsr.pt")
print(tnsr.shape)
img = transform(tnsr[0,:,:,:])
img.show()

    # "/groups/claytonm/gauravs_data/clevrmath_data/"