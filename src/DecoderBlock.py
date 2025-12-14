import torch
import torch.nn as nn
from src.Config import Config

class TransformerBlock(nn.Module):
    """
    A single block of the Transformer decoder.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.device = config.device
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        # Feed-forward network dimension
        
        # standard is 4 times the model dimension
        self.ffn_dim = 4 * self.model_dim
        self.dropout_rate = config.dropout

        # Precompute the causal mask
        self.causal_mask = self.generate_causal_mask(config.N)

        # layer norm pre version.
        self.ln1 = nn.LayerNorm(self.model_dim)
        # TODO: implement multi-head self-attention
        self.attn = nn.MultiheadAttention(self.model_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True)
        self.ln2 = nn.LayerNorm(self.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.ffn_dim),
            nn.GELU(), # or nn.RELU() ? Relu should work best for deeper networks
            nn.Linear(self.ffn_dim, self.model_dim),
            nn.Dropout(self.dropout_rate)
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, attn_mask=None):
        """Causal self-attention block with feed-forward network."""
        if attn_mask is None:
            attn_mask = self.causal_mask
        # Self-attention with residual
        # query, key, value are all attending to x in self-attention
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0])
        # Feed-forward with residual
        x = x + self.mlp(self.ln2(x))
        return x
    
    def generate_causal_mask(self, size):
        """Generate a causal mask for self-attention."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.to(device=self.device)
        return mask