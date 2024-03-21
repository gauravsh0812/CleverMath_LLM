import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, UNET, ROBERTA):
        super(ClevrMath_model, self).__init__()
        self.unet = UNET
        self.roberta = ROBERTA

    def forward(self, imgs, ids, attns):
        encoded_imgs = self.unet(imgs)
        encoded_qtns = self.roberta(ids, attns)

