import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 encoder, 
                 vit_encoder,
                 decoder,
                 encadaptor,
                 robertaadaptor,
                 projector,
    ):
        super(ClevrMath_model, self).__init__()
        self.enc = encoder
        self.vit_encoder = vit_encoder
        self.encadaptor = encadaptor
        self.robertaadaptor = robertaadaptor
        self.dec = decoder
        self.projector = projector

    def forward(self, imgs, ids, attns, device):
        masks = self.enc(imgs)  # (B, top_n, w,h)
        vit_masks = self.vit_encoder(masks) # (B, n_patch, emd_dim)
        vit_imgs = self.vit_encoder(imgs)   # (B, n_patch, emb_dim)
        vit_output = torch.cat((vit_masks, vit_imgs), dim=-1) # (B, n_patches, 2*emb_dim)
        encoutput = self.encadaptor(vit_output)  # (B, max_len, 64)

        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        
        roboutput = self.robertaadaptor(last_hidden_roberta) # (B, max, 64)
        output = self.projector(encoutput, roboutput) # (B,11)
        
        return output