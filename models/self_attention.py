import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Self_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (batch_size, seq_length, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)  # (batch_size, seq_length, seq_length)
        
        # Compute the weighted sum of values
        attention_output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, embed_dim)
        
        return attention_output