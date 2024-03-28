import torch.nn as nn
import torch
from transformers import LlamaForSequenceClassification

class Llama2Decoder(nn.Module):
    def __init__(self,):
        super(Llama2Decoder, self).__init__()
        self.model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")

    def forward(self, x):
        # x: (B, seq_len), dtype=long
        output = self.model.generate(input_ids=x,
                                     output_hidden_states=True)

        print(output.shape)

l = Llama2Decoder()
x = torch.rand((10, 64))
l(x)