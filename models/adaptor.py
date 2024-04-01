import torch 
import torch.nn as nn

class Adaptor(nn.Module):
    def __init__(self, in_dim, features, num_classes):
        super(Adaptor, self).__init__()

        # features: [512, 256, 128, 64]
        self.lin1 = nn.Linear(in_dim, features[0])
        self.lin2 = nn.Linear(features[0], features[1])
        self.lin3 = nn.Linear(features[1], features[2])
        self.lin4 = nn.Linear(features[2], features[3])
        self.lin5 = nn.Linear(features[3], num_classes)
        self.final = nn.Linear(num_classes*2, num_classes)
        self.proj_clip = nn.Linear(50,19)
        self.proj_final = nn.Linear(19,num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x_clip, x_roberta):

        xc = self.proj_clip(x_clip.permute(0,2,1)).permute(0,2,1)
        xc = self.relu(self.lin1(xc))
        xc = self.relu(self.lin2(xc))
        xc = self.relu(self.lin3(xc))
        xc = self.relu(self.lin4(xc))
        xc = self.relu(self.lin5(xc))
        
        xr = self.relu(self.lin1(x_roberta))
        xr = self.relu(self.lin2(xr))
        xr = self.relu(self.lin3(xr))
        xr = self.relu(self.lin4(xr))
        xr = self.relu(self.lin5(xr))
        
        # x_roberta + x
        x = torch.cat((xc,xr), dim=-1)
        x = self.relu(self.final(x))
        x = self.proj_final(x.permute(0,2,1)).permute(0,2,1)
        return x   # (B, 11, 11)