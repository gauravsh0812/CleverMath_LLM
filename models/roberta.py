from transformers import RobertaTokenizer


class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    

    def forward(self, texts):
        pass

    tokenizer("Hello world")["input_ids"]

    tokenizer(" Hello world")["input_ids"]