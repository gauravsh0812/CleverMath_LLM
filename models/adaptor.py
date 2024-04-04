import torch 
import torch.nn as nn

class ClipAdaptor(nn.Module):
    def __init__(self, clip_in_dim, roberta_in_dim, max_len):
        super(ClipAdaptor, self).__init__()

        self.cliplin1 = nn.Linear(clip_in_dim, roberta_in_dim)
        self.proj_clip = nn.Linear(50,max_len)
        self.relu = nn.ReLU()
    
    def forward(self, x_clip, x_roberta):

        xc = self.proj_clip(x_clip.permute(0,2,1)).permute(0,2,1)
        xc = self.relu(self.cliplin1(xc))
        
        # x_roberta + x
        x = torch.cat((xc,x_roberta), dim=-1)  
        return x   # (B, max_len, 768)

class Projector(nn.Module):

    def __init__(self, roberta_in_dim, features, max_len, num_classes):
        super(Projector, self).__init__()
        self.lin1 = nn.Linear(roberta_in_dim, features[0])
        self.lin2 = nn.Linear(features[0], features[1])
        self.lin3 = nn.Linear(features[1], features[2])
        self.lin4 = nn.Linear(features[2], features[3])
        self.lin5 = nn.Linear(features[3], num_classes)
        self.final_lin = nn.Linear(max_len*num_classes, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, xc):
        xc = self.relu(self.lin1(xc))
        xc = self.relu(self.lin2(xc))
        xc = self.relu(self.lin3(xc))
        xc = self.relu(self.lin4(xc))
        xc = self.relu(self.lin5(xc))  # (B, max, 11)
        xc = torch.flatten(xc, start_dim=-2, end_dim=-1)
        xc = self.relu(self.final_lin(xc))
        return xc   # (B, 11)