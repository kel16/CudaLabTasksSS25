import torch.nn as nn

from models.MLP import MLP
from models.MultiHeadSelfAttention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Transformer block using self-attention

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, token_dim, attn_dim, num_heads, mlp_size):
        """ Module initializer """
        super().__init__()
        self.token_dim = token_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # MHA
        self.ln_att = nn.LayerNorm(token_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
                token_dim=token_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            )
        
        # MLP
        self.ln_mlp = nn.LayerNorm(token_dim, eps=1e-6)
        self.mlp = MLP(
                in_dim=token_dim,
                hidden_dim=mlp_size,
            )
        return


    def forward(self, inputs):
        """
        Forward pass through transformer encoder block.
        We assume the more modern PreNorm design
        """
        assert inputs.ndim == 3

        # Self-attention.
        x = self.ln_att(inputs)
        x = self.attn(x)
        y = x + inputs

        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y

        return z


    def get_attention_masks(self):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.attention_map
        N = attn_masks.shape[-1]
        attn_masks = attn_masks.reshape(-1, self.num_heads, N, N)
        return attn_masks
    