import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 encoder, 
                 decoder,
                 adaptor,

    ):
        super(ClevrMath_model, self).__init__()
        self.enc = encoder
        self.adaptor = adaptor
        self.dec = decoder

    def forward(self, imgs, ids, attns, device):
        encoded_imgs,pooled_layers = self.enc(imgs, device)  # (B, L=w*h, dim)
        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        
        output = self.adaptor(encoded_imgs,
                            last_hidden_roberta)  # (B, max_len, num_classes)
        
        return output