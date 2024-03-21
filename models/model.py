import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 UNET, 
                 ROBERTA,
                 features):
        super(ClevrMath_model, self).__init__()
        self.unet = UNET
        self.roberta = ROBERTA
        self.proj = nn.Linear(features[0], 768)

    def forward(self, imgs, ids, attns):
        encoded_imgs = self.unet(imgs).permute(1,0,2)  # (B, L=w*h, features[0])
        last_hidden_roberta = self.roberta(ids, attns) # (B, L_question, 768)

        # project the outputs 
        encoded_imgs = self.proj(encoded_imgs) # (B, L, 768)
        encoded_imgs = encoded_imgs.permute(0,2,1) # (B, 768, L)
        last_hidden_roberta = last_hidden_roberta.permute(0,2,1) # (B, 768, L_qtn)

        # concat
        output = torch.cat((encoded_imgs, last_hidden_roberta), dim=2)
        return output