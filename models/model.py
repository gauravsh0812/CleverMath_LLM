import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self,  
                 vit_encoder,
                 decoder,
                 encadaptor,
                 robertaadaptor,
                 projector,
                 top_n,
                 mask_path,
    ):
        super(ClevrMath_model, self).__init__()
        self.vit_encoder = vit_encoder
        self.encadaptor = encadaptor
        self.robertaadaptor = robertaadaptor
        self.dec = decoder
        self.projector = projector
        self.top_n = top_n
        self.mask_path = f"{mask_path}/maskrcnn/masks"
        self.score_path = f"{mask_path}/maskrcnn/scores"

    def get_masks(self,imgs):
        
        final_masks = []

        for im in imgs:
            im = int(im.item())
            _mask = torch.load(f"{self.mask_path}/{im}.pt")
            _score = torch.load(f"{self.score_path}/{im}.pt").tolist()

            top_masks = []

            if _mask.shape[0] >= self.top_n:
                top_n_scores = sorted(_score, reverse=True)[:self.top_n]
                for _s in top_n_scores:
                    top_n_index = _score.index(_s)
                    top_n_mask = _mask[top_n_index,:,:,:] #(1,1,w,h)
                    top_masks.append(top_n_mask)
                masks = torch.stack(top_masks).squeeze(1)  # (top_n,w,h)
                final_masks.append(masks)

            else:
                delta = self.top_n - _mask.shape[0] 
                top_n_scores = sorted(_score, reverse=True)[:self.top_n]
                for _s in top_n_scores:
                    top_n_index = _score.index(_s)
                    top_n_mask = _mask[top_n_index,:,:,:] #(1,1,w,h)
                    top_masks.append(top_n_mask)      

                masks = torch.stack(top_masks).squeeze(1)  # (mask_shape[0], 1, w,h)          
                zeros = torch.zeros(delta,masks.shape[-2],masks.shape[-1])  
                masks = torch.cat((masks, zeros), dim=0)  #(top_n,1,w,h)
                final_masks.append(masks)
            
        final_masks = torch.stack(final_masks)
        return final_masks # (B, top_n, w,h)


    def forward(self, imgs, img_tnsrs, ids, attns, device):
        masks = self.get_masks(imgs).to(device)  # (B, top_n, w,h)
        vit_masks = self.vit_encoder(masks) # (B, n_patch, emd_dim)
        print("vit mask shape: ", vit_masks.shape)
        vit_imgs = self.vit_encoder(img_tnsrs)   # (B, n_patch, emb_dim)
        print("vit imgs shape: ", vit_imgs.shape)
        vit_output = torch.cat((vit_masks, vit_imgs), dim=-1) # (B, n_patches, 2*emb_dim)
        encoutput = self.encadaptor(vit_output)  # (B, max_len, 64)

        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        
        roboutput = self.robertaadaptor(last_hidden_roberta) # (B, max, 64)
        output = self.projector(encoutput, roboutput) # (B,11)
        
        return output
    