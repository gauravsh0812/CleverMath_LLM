import torch.nn as nn
import torch
from torchvision import models

# Load the pre-trained EfficientNet model
model = models.efficientnet_b0(pretrained=True)

resnet18 = models.resnet18(pretrained=True)
encoded_img = nn.Sequential(*(list(resnet18.children())[:-2]))


# for i in range():
i = 9207
f = torch.load(f"/groups/claytonm/gauravs_data/clevrmath_data/image_tensors/{i}.pt")[:3,:,:]
f = f.unsqueeze(0)

features = model.features(f)
print(features.shape)
if torch.isnan(features).any():
    print("features contains NaN:", torch.isnan(features).any())

features = encoded_img(f)
print(features.shape)
if torch.isnan(features).any():
    print("features contains NaN:", torch.isnan(features).any())