from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    

    def forward(self, qtns):
        print(qtns.shape)