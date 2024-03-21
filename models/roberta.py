from transformers import RobertaModel
import torch.nn as nn

class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    

    def forward(self, ids, attns):
        print("roberta: ", ids.shape, attns.shape)
        outputs = self.model(input_ids=ids,
                             attention_mask=attns)
        last_hidden_states = outputs.last_hidden_state
        print("roberta last hid shape: ", last_hidden_states.shape)