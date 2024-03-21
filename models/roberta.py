from transformers import RobertaModel
import torch.nn as nn

class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    

    def forward(self, qtns):
        print("roberta: ", qtns.shape)