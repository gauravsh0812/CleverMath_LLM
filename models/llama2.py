import torch.nn as nn
import torch
from transformers import LlamaModel


class Llama2Decoder(nn.Module):
    def __init__(self,api_key):
        super(Llama2Decoder, self).__init__()

        self.model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                                    token=api_key)

    def forward(self, x):
        # x: (B, seq_len), dtype=long
        output = self.model(input_ids=x,
                            output_hidden_states=True)

        print(output.shape)

l = Llama2Decoder("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")
x = torch.rand(1, 64).long()
l(x)