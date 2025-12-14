import torch
import torch.nn as nn
from src.Config import Config
from src.MultiHeadAttention import MultiHeadAttention

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
        # Register as buffer so it moves to device automatically with the model
        self.register_buffer('causal_mask', self.generate_causal_mask(config.N))

        # layer norm pre version.
        self.ln1 = nn.LayerNorm(self.model_dim)
        # TODO: implement multi-head self-attention
        # self.attn = nn.MultiheadAttention(self.model_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True)
        
        # Custom Multi-Head Attention
        self.attn = MultiHeadAttention(config)
        
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
            # Slice the mask to the current sequence length (zero-copy view)
            seq_len = x.size(1)
            attn_mask = self.causal_mask[:seq_len, :seq_len]
            
        # Self-attention with residual
        # query, key, value are all attending to x in self-attention
        # Note: self.attn returns (output, weights), we take [0]
        
        attn_out, _ = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x = x + self.mlp(self.ln2(x))
        return x
    
    def generate_causal_mask(self, size):
        """Generate a causal mask for self-attention."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        # mask = mask.to(device=self.device) # Handled by register_buffer
        # Invert mask for masked_fill (True means mask out / -inf)
        # But standard PyTorch attention often uses True to KEEP or False to MASK depending on implementation.
        # In my implementation above: scores.masked_fill(attn_mask == 0, float('-inf'))
        # So we need 1s where we attend, 0s where we mask.
        # triu(diagonal=1) gives 1s in upper triangle (future). We want to mask those.
        # So we want 1s in lower triangle.
        return ~mask