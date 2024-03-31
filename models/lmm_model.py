import torch 
import torch.nn as nn

class LLM_model(nn.Module):

    def __init__(self, 
                 encoder, 
                 decoder1,
                 decoder2,
                 adaptor,
                 dim,
                 image_length,
                 max_len,
                 num_classes,
    ):
        super(LLM_model, self).__init__()
        self.enc = encoder
        self.adaptor = adaptor
        self.dec1 = decoder1
        self.dec2 = decoder2
        self.proj1 = nn.Linear(dim, 768)
        self.proj2 = nn.Linear(image_length, max_len)
        self.clf1 = nn.Linear(64, num_classes)
        self.clf2 = nn.Linear(max_len, num_classes)

    def forward(self, imgs, ids, attns):
        encoded_imgs = self.enc(imgs)  # (B, L=w*h, dim)
        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        
        output = self.adaptor(encoded_imgs,
                                    last_hidden_roberta)  # (B, features[-1])
        # classifier
        output = self.clf1(output)
        output = self.clf2(output.permute(0,2,1)).permute(0,2,1) # (B, num_classes, num_classes)

        return output