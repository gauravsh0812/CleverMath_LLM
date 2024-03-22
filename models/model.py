import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 UNET, 
                 ROBERTA,
                 features,
                 image_length,
                 max_len,
    ):
        super(ClevrMath_model, self).__init__()
        self.unet = UNET
        self.roberta = ROBERTA
        self.proj1 = nn.Linear(features[0], 768)
        self.proj2 = nn.Linear(image_length, max_len)

    def forward(self, imgs, ids, attns):
        encoded_imgs = self.unet(imgs).permute(1,0,2)  # (B, L=w*h, features[0])
        last_hidden_roberta = self.roberta(ids, attns) # (B, max_len, 768)

        # project the outputs 
        encoded_imgs = self.proj1(encoded_imgs) # (B, L, 768)
        encoded_imgs = encoded_imgs.permute(0,2,1) # (B, 768, L)
        encoded_imgs = self.proj2(encoded_imgs) # (B, 768, max_len)
        encoded_imgs = encoded_imgs.permute(0,2,1) # (B, max_len, 768)

        # concat
        output = torch.cat((encoded_imgs, last_hidden_roberta), dim=2) # (B, max_len, 768*2)

        # classifier
        output = torch.flatten(output, start_dim=2, end_dim=-1)  # (B, -1)

        print(output.shape)

        return output