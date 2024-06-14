import torch.nn as nn
import torch
from torchvision import models

# Load the pre-trained EfficientNet model
model = models.efficientnet_b0(pretrained=True)

resnet18 = models.resnet18(pretrained=True)
encoded_img = nn.Sequential(*(list(resnet18.children())[:-2]))

f = torch.load("/groups/claytonm/gauravs_data/clevrmath_data/image_tensors/0.pt")[:3,:,:]
f = f.unsqueeze(0)

features = model.features(f)
print(features.shape)
if torch.isnan(features).any():
    print("features contains NaN:", torch.isnan(features).any())

features = encoded_img.features(f)
print(features.shape)
if torch.isnan(features).any():
    print("features contains NaN:", torch.isnan(features).any())