from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel

class ClipVisionEncoder(nn.Module):
    
    def __init__(self,):
        super(ClipVisionEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid, _pool = list(), list()

        print("image_paths: ", image_paths)

        for image_path in image_paths:

            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled classes states
            
            print("clip: ", last_hidden_state.shape, pooled_output.shape)

            _hid.append(last_hidden_state.unsqueeze(0))
            _pool.append(pooled_output.unsqueeze(0))

        # hidden: (B, L, 768)
        # pooled: (B, 768)
        return torch.stack(_hid).to(device), torch.stack(_pool).to(device)
    
# cve = ClipVisionEncoder()
# cve(["data/images/0.png","data/images/1.png"],
#      "cpu")