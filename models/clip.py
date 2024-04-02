from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig

class ClipVisionEncoder(nn.Module):
    
    def __init__(self, finetune=False, config=None):
        super(ClipVisionEncoder, self).__init__()
        if finetune:
            configuration = CLIPVisionConfig(**config)
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPVisionModel(configuration)
        else:
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid, _pool = list(), list()
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled classes states

            _hid.append(last_hidden_state.squeeze(0))
            _pool.append(pooled_output.squeeze(0))

        # hidden: (B, L, 768)
        # pooled: (B, 768)
        return torch.stack(_hid).to(device), torch.stack(_pool).to(device)

configuration={
      "hidden_size":512,
      "intermediate_size": 1024,
      "projection_dim": 512,
      "num_hidden_layers": 6,
      "num_attention_heads": 8,
      "num_channels": 3,
      "image_size": 224,
      "patch_size": 32
      }

cve = ClipVisionEncoder(finetune=True, config=configuration).to("cuda:0")
cve(["/groups/claytonm/gauravs_data/clevrmath_data/data/images/13704.png"], "cuda:0")