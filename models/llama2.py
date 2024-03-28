import torch.nn as nn
import torch
from transformers import LlamaForSequenceClassification

class Llama2Decoder(nn.Module):
    def __init__(self,):
        super(Llama2Decoder, self).__init__()

        self.model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                                    token="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")

    def forward(self, x):
        # x: (B, seq_len), dtype=long
        output = self.model(input_ids=x,
                                     output_hidden_states=True)

        print(output.shape)

l = Llama2Decoder()
# x = torch.rand(10, 64).long()
x = [torch.rand(64).long() for _ in range(10)]
for _x in x:
    l(_x.unsqueeze(0))