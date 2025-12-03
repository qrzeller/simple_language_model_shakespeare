'''
Docstring for src.Transformer_Decoder
This module implements the Transformer Decoder architecture for sequence modeling tasks.
As per requested by the miniproject assignement.
Author: Quentin Zeller
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder model for sequence modeling.
    """

    def __init__(self, config):
        super(TransformerDecoder, self).__init__() # needed to initialize nn.Module (ex: instantiate layers or move to device)
        
        # get the hyperparameters from config
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length

        # TODO: Should we do embedding ourselves or use nn.Embedding?
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.model_dim)


    def forward(self, x, tgt_mask=None):
        
        return logits