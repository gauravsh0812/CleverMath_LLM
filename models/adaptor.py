import torch 
import torch.nn as nn

class Adaptor(nn.Module):
    def __init__(self, in_dim, features):
        super(Adaptor, self).__init__()

        # features: [512, 264, 128, 64]
        self.lin1 = nn.Linear(in_dim, features[0])
        self.lin2 = nn.Linear(features[0], features[1])
        self.lin3 = nn.Linear(features[1], features[2])
        self.relu = nn.ReLU()
    
    def forward(self, x_clip, x_roberta):
        xc = self.relu(self.lin1(x_clip))
        xc = self.relu(self.lin2(xc))
        xc = self.relu(self.lin3(xc))

        xr = self.relu(self.lin1(x_roberta))
        xr = self.relu(self.lin2(xr))
        xr = self.relu(self.lin3(xr))

        # x_roberta + x
        x = torch.cat((xc,xr), dim=0)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        
        return x   # (B, features[-1])
