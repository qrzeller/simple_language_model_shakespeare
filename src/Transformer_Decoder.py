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
from DecoderBlock import TransformerBlock


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
        self.config = config # to pass it all further

        # TODO: Should we do embedding ourselves or use nn.Embedding?
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dim)

        # learned positional embeddings is also possible :
        #self.position_embedding = nn.Embedding(self.max_seq_length, self.model_dim)
        # fourier features as in "Attention is all you need"
        self.position_encoding = PositionalEncoding(self.model_dim)
        
        # TODO: Try this model
        # Transformer decoder blocks, different weight matrices for each layer
        # self.decoder_blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.num_layers)])

        # or a single decoder block reused multiple times (weight sharing)
        self.decoder_block = TransformerBlock(self.config)

        # Alternative is to use the first layer as independent block then weight share the rest
        # That's what we do in some cross attention models

        # Projection to final classes (vocab size)
        self.output_projection = nn.Linear(self.model_dim, self.vocab_size)

    def forward(self, x, tgt_mask=None):
        """
        Forward pass of the Transformer decoder.

        Args:
            x (torch.LongTensor): Input token indices of shape (batch_size, seq_len).
            tgt_mask (torch.BoolTensor, optional): 2D attention mask of shape (seq_len, seq_len).
                Mask value True means "do not attend" for that (query, key) pair.
                If None, a **causal (future-masking) mask** is used: positions cannot attend to tokens to their right
                (upper triangular mask above the main diagonal). The dtype should be bool and the tensor must be on
                the same device as the inputs.

        Returns:
            logits (torch.FloatTensor): shape (batch_size, seq_len, vocab_size), unnormalized scores.
        """
        # noteboox pseudocode reference : WTE(idx)
        token_embeddings = self.token_embedding(x)  # (seq_len, batch_size, model_dim)
        token_embeddings = token_embeddings.transpose(0, 1)  # Transformer expects (seq_len, batch_size, model_dim)
        # noteboox pseudocode reference : WPE(pos)
        x = self.position_encoding(token_embeddings)  # Add positional encoding
        # it should be just a view
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, model_dim)
        # TODO: verify shape correctness
        
        # From pseudocode from notebook : x = Dropout(tok_emb + pos_emb)
        x = F.dropout(x, p=0.1, training=self.training)

        # Pass through Transformer decoder layers
        for _ in range(self.num_layers):
            x = self.decoder_block(x, attn_mask=tgt_mask)
        
        # Final linear layer to project to vocabulary size
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
