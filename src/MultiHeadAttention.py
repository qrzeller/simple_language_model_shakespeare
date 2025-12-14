import torch
import torch.nn as nn
from src.Config import Config


""" Multi-Head Attention module.

As per Wikipedia:

Multi-head attention

Decoder multiheaded cross-attention
Multi-head attention
MultiHead(Q, K, V) = Concat(head1, ..., headh) WO
{\displaystyle {\text{MultiHead}}(Q,K,V)={\text{Concat}}({\text{head}}_{1},...,{\text{head}}_{h})W^{O}}
where each head is computed with QKV attention as:
headi = Attention(QWiQ, KWiK, VWiV)
{\displaystyle {\text{head}}_{i}={\text{Attention}}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})}
and WiQ, WiK, WiV {\displaystyle W_{i}^{Q},W_{i}^{K},W_{i}^{V}}, and WO {\displaystyle W^{O}} are parameter matrices.

The permutation properties of (standard, unmasked) QKV attention apply here also. For permutation matrices, A, B
{\displaystyle A,B}:
MultiHead(AQ, BK, BV) = A MultiHead(Q, K, V)
{\displaystyle {\text{MultiHead}}(AQ,BK,BV)=A\,{\text{MultiHead}}(Q,K,V)}

From which we also see that multi-head self-attention:
X \mapsto MultiHead(XTq, XTk, XT v)
{\displaystyle X\mapsto {\text{MultiHead}}(XT_{q},XT_{k},XT_{v})}
is equivariant with respect to re-ordering of the rows of input matrix X {\displaystyle X}.
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.device = config.device
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        
        assert self.model_dim % self.num_heads == 0, "model_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        # we can instantiate the full contiguous matrix and split later
        self.W_Q = nn.Linear(self.model_dim, self.model_dim)
        # Eventually share weights for K and V ?
        self.W_K = nn.Linear(self.model_dim, self.model_dim)
        self.W_V = nn.Linear(self.model_dim, self.model_dim)
        
        # Final linear projection
        self.W_O = nn.Linear(self.model_dim, self.model_dim)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key=None, value=None, attn_mask=None):

        # Preserve the possibility to have different key, value (for cross-attention)
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.size(0)
        seq_len = query.size(1)

        # 1. Linear Projections
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # 2. Split heads and reshape
        # (batch * seq_len, model_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        # we want the head first so we handle better the spilt/concat operations
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # following the formula : scores = QK^T / sqrt(d_k)
        # We need to normalize by the dimension to preserve the variance, mainly for the stability (gradient, precision, ..., probably reuse of weights)
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len) -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            # attn_mask is (seq_len, seq_len), broadcast to (batch, num_heads, seq_len, seq_len)
            # -inf instead of 0 for proper masking in softmax, because e^-inf = 0
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted sum of values
        # following the formula: context = softmax(QK^T / sqrt(d_k)) V
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, V)

        # 5. Concatenate heads
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, model_dim)
        # contiguous to ensure memory layout, necessary after transpose before view
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

        # 6. Final linear projection
        output = self.W_O(context)
        
        return output, attn_weights