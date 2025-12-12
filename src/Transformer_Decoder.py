'''
Docstring for src.Transformer_Decoder
This module implements the Transformer Decoder architecture for sequence modeling tasks.
As per requested by the miniproject assignement.
Author: Quentin Zeller
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config
from positional_encoding import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder model for sequence modeling.
    """

    def __init__(self, config: Config):
        super(TransformerDecoder, self).__init__() # needed to initialize nn.Module (ex: instantiate layers or move to device)
        
        # get the hyperparameters from config
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length

        # TODO: Should we do embedding ourselves or use nn.Embedding?
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dim)

        # learned positional embeddings is also possible :
        #self.position_embedding = nn.Embedding(self.max_seq_length, self.model_dim)
        # fourier features as in "Attention is all you need"
        self.position_encoding = PositionalEncoding(self.model_dim)


    def forward(self, x, tgt_mask=None):
        # noteboox pseudocode reference : WTE(idx)
        token_embeddings = self.token_embedding(x)  # (seq_len, batch_size, model_dim)
        token_embeddings = token_embeddings.transpose(0, 1)  # Transformer expects (seq_len, batch_size, model_dim)
        # noteboox pseudocode reference : WPE(pos)
        x = self.position_encoding(token_embeddings)  # Add positional encoding
        # it should be just a view
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, model_dim)
        
        # From pseudocode from notebook : x = Dropout(tok_emb + pos_emb)
        x = F.dropout(x, p=0.1, training=self.training)

        return logits
    
