# We input to transformer a vector + it's embedding
# Since it is GPT style architecture, the fourrier features are added and not concatenated
import torch
import torch.nn as nn
import math
from src.Config import Config


class PositionalEncoding(nn.Module):
    # Max_len should be the size of the maximum context length seen
    # If bidirectional, i assume we would need N*2
    def __init__(self, d_model, config: Config):
        max_len = config.N

        super(PositionalEncoding, self).__init__()
        # malloc the positional encodings once
        pe = torch.zeros(max_len, d_model)
        # arange is flattened vector of positions
        # unsqueeze(1) adds a dimension at index 1, for later broadcasting
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # from attention is all you need paper, 1000 fits well conventional hardware
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # usefull for the model to learn relative positions
        # ex : sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # so we can add it to the token embeddings later
        # because x is of shape (seq_len, batch_size, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # since it's not learned, optimize it out of the model parameters
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add the positional encoding to the input embeddings
        # doesnt bother so much text while it could be an issue for other tasks !
        # in which case, concatenation could be better
        x = x + self.pe[:x.size(0), :]
        return x