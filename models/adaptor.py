import torch 
import torch.nn as nn

class Adaptor(nn.Module):
    def __init__(self, in_dim, features):
        super(Adaptor, self).__init__()

        self.lin1 = nn.Linear(in_dim, features[0])
        self.lin2 = nn.Linear(features[0], features[1])
        self.lin3 = nn.Linear(features[1], features[2])
        self.relu = nn.ReLU()
    
    def forward(self, x_clip, x_roberta):
        x = self.relu(self.lin1(x_clip))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        # x_roberta + x
        torch.

        return x