import torch, os
import torch.nn as nn
from PIL import Image
import yaml
from box import Box

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModel, 
    RobertaModel,
)

with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

class ClipVisionEncoder(nn.Module):
    
    def __init__(self,):
        super(ClipVisionEncoder, self).__init__()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid = list()
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
            _hid.append(last_hidden_state.squeeze(0))
        
        # hidden: (B, L, 768)
        return torch.stack(_hid).to(device)#, torch.stack(_pool).to(device)

def lisa(imgs):
    tnsrs = []
    for i in imgs:
        i = os.path.basename(i).split(".")[0]
        tnsr = torch.load(f"{cfg.dataset.path_to_data}/lisa/masked_image_tensors/{i}.pt")
        tnsrs.append(tnsr)
    
    return torch.stack(tnsrs)

class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")        

    def forward(self, ids, attns):
        # shape of ids and attns: (B, max_len)
        outputs = self.model(input_ids=ids,
                             attention_mask=attns)
        last_hidden_states = outputs.last_hidden_state # (B, max_len, 768)
        return last_hidden_states

class ClipAdaptor(nn.Module):
    def __init__(self, clip_in_dim, features, max_len):
        super(ClipAdaptor, self).__init__()

        self.cliplin1 = nn.Linear(clip_in_dim, features[0])
        self.cliplin2 = nn.Linear(features[0], features[1])
        self.cliplin3 = nn.Linear(features[1], features[2])
        self.cliplin4 = nn.Linear(features[2], features[3])
        self.proj_clip = nn.Linear(50,max_len)
        self.relu = nn.ReLU()
    
    def forward(self, xc):
        
        xc = self.relu(self.cliplin1(xc))
        xc = self.relu(self.cliplin2(xc))
        xc = self.relu(self.cliplin3(xc))
        xc = self.relu(self.cliplin4(xc))
        xc = self.relu(self.proj_clip(xc.permute(0,2,1))).permute(0,2,1)
        
        return xc # (B,max_len, 64)

class RobertaAdaptor(nn.Module):
    def __init__(self, roberta_in_dim, features):
        super(RobertaAdaptor, self).__init__()

        self.roblin1 = nn.Linear(roberta_in_dim, features[0])
        self.roblin2 = nn.Linear(features[0], features[1])
        self.roblin3 = nn.Linear(features[1], features[2])
        self.roblin4 = nn.Linear(features[2], features[3])
        self.gelu = nn.GELU()
    
    def forward(self, xr):
        xr = self.gelu(self.roblin1(xr))
        xr = self.gelu(self.roblin2(xr))
        xr = self.gelu(self.roblin3(xr))
        xr = self.gelu(self.roblin4(xr))
        
        return xr # (B,max_len, 64)

class LisaAdaptor(nn.Module):
    def __init__(self, lisa_in_dim, features, max_len):
        super(LisaAdaptor, self).__init__()

        self.lisalin1 = nn.Linear(lisa_in_dim, features[0])
        self.lisalin2 = nn.Linear(features[0], features[1])
        self.lisalin3 = nn.Linear(features[1], features[2])
        self.lisalin4 = nn.Linear(features[2], features[3])
        self.proj_lisa = nn.Linear(50,max_len)
        self.relu = nn.ReLU()
    
    def forward(self, xc):
        
        xc = self.relu(self.lisalin1(xc))
        xc = self.relu(self.lisalin2(xc))
        xc = self.relu(self.lisalin3(xc))
        xc = self.relu(self.lisalin4(xc))
        xc = self.relu(self.proj_lisa(xc.permute(0,2,1))).permute(0,2,1)
        
        return xc # (B,max_len, 64)

class Projector(nn.Module):

    def __init__(self, features, max_len, num_classes):
        super(Projector, self).__init__()
        if self.fusion == "concat":
            self.final_lin1 = nn.Linear(max_len*2, max_len)
            self.attn = Self_Attention(features[-1])
            self.final_lin3 = nn.Linear(features[-1], num_classes)
            self.norm = nn.BatchNorm1d(features[-1])
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.gelu = nn.GELU()
    
    def combine(self,xc,xr):
        x = torch.cat((xc,xr), dim=1)  
        x = self.gelu(self.norm(self.final_lin1(x.permute(0,2,1)))).permute(0,2,1)
        x = self.attn(x)  # (B, max, 64)
        return x

    def forward(self, xc, xr, pool=False):
        x = self.combine(xc,xr)
        
        if pool:
            x = self.pool(x.permute(0,2,1)).permute(0,2,1)  # (B, max_len=1, 64)
            x = torch.flatten(x, -2,-1)   # (B,max_len*64) >> (B, 64)
            x = self.gelu(self.final_lin3(x))  # (B, num_classes)
        
        return x   # (B, max, 64) or if pool then (B,num_classes)

class Self_Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Self_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (batch_size, seq_length, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)  # (batch_size, seq_length, seq_length)
        
        # Compute the weighted sum of values
        attention_output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, embed_dim)
        
        return attention_output       

class ClevrMath_model(nn.Module):

    def __init__(self, max_len, ans_vocab):

        super(ClevrMath_model, self).__init__()
        self.clipenc = ClipVisionEncoder()
        self.robenc = RobertaEncoder()
        self.lisaadaptor = LisaAdaptor(
                                480,
                                cfg.training.adaptor.features,
                                max_len,
                            )
        self.clipadaptor = ClipAdaptor(
                                768, 
                                cfg.training.adaptor.features,
                                max_len,
                            )
        self.robertaadaptor = RobertaAdaptor(
                                cfg.training.roberta.in_dim,
                                cfg.training.adaptor.features,
                            )
        self.projector = Projector(
                                cfg.training.adaptor.features,
                                max_len, 
                                len(ans_vocab),
                            )
        for param in self.clipenc.parameters():
            param.requires_grad = False

        for param in self.robenc.parameters():
            param.requires_grad = False

    def forward(
            self, 
            imgs,
            qtn_ids,
            qtn_attns,
            device,
        ):
        
        lisa_tnsr = lisa(imgs)   # (B, 320, 480)
        encoded_imgs = self.clipenc(imgs, device)  # (B, L=w*h, dim)
        last_hidden_roberta = self.robenc(qtn_ids, qtn_attns) # (B, max_len, 768)   

        lisaoutput = self.lisaadaptor(lisa_tnsr)    # (B, max_len, 64) 
        clipoutput = self.clipadaptor(encoded_imgs)  # (B, max_len, 64)
        visionoutput = self.projector(lisaoutput, clipoutput)

        roboutput = self.robertaadaptor(last_hidden_roberta) # (B, max, 64)
        projoutput = self.projector(visionoutput, roboutput, pool=True) # (B,num_classes)
        
        return projoutput