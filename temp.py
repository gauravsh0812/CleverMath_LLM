import torch.nn as nn
import torch
from torchvision import models

# Load the pre-trained EfficientNet model
model = models.efficientnet_b0(pretrained=True)

f = torch.load("/groups/claytonm/gauravs_data/clevrmath_data/image_tensors/0.pt")[:3,:,:]
f = f.unsqueeze(0)

features = model.features(f)
print(features.shape)