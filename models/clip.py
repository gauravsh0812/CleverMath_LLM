from PIL import Image
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel

class ClipVisionEncoder(nn.Module):
    
    def __init__(self,):
        super(ClipVisionEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled classes states

        return last_hidden_state, pooled_output


c = ClipVisionEncoder()
h,p = c("0.png")
print("h: ", h.shape)
print("p: ", p.shape)