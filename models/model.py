import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 encoder, 
                 decoder,
                 dim,
                 image_length,
                 max_len,
                 num_classes,
    ):
        super(ClevrMath_model, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.proj1 = nn.Linear(dim, 768)
        self.proj2 = nn.Linear(image_length, max_len)
        self.clf1 = nn.Linear(768*2, num_classes)
        self.clf2 = nn.Linear(max_len, num_classes)

    def forward(self, imgs, ids, attns):
        encoded_imgs = self.enc(imgs).permute(1,0,2)  # (B, L=w*h, dim)
        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        

        # project the outputs 
        encoded_imgs = self.proj1(encoded_imgs) # (B, L, 768)
        encoded_imgs = encoded_imgs.permute(0,2,1) # (B, 768, L)
        encoded_imgs = self.proj2(encoded_imgs) # (B, 768, max_len)
        encoded_imgs = encoded_imgs.permute(0,2,1) # (B, max_len, 768)

        # concat
        output = torch.cat((encoded_imgs, last_hidden_roberta), dim=2) # (B, max_len, 768*2)

        # classifier
        output = self.clf1(output)
        output = self.clf2(output.permute(0,2,1)).permute(0,2,1) # (B, num_classes, num_classes)

        return output