import flax.linen as nn
import jax.numpy as jnp
from attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: [ batch_size, seq_len, embed_dim ]
        
        # 1. Layer Normalization 1 & Causal Self Attention (with Skip Connection)
        # We normalize the inputs to keep coordinate values in a stable, healthy range.
        norm_x = nn.LayerNorm()(x)
        attn_out = CausalSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)(norm_x)
        x = x + attn_out
        
        # 2. Layer Normalization 2 & MLP (with Skip Connection)
        # Normalize coordinates again before individual token processing
        norm_x2 = nn.LayerNorm()(x)
        
        # 3. MLP: Let each token process its own data in isolation
        # - Expand 128 coordinates to 512 to allow more workspace
        # - Apply non-linear GELU gate so model can learn complex mathematical logic
        # - Compress back to 128 coordinates
        h = nn.Dense(features=512)(norm_x2)
        h = nn.gelu(h)
        mlp_out = nn.Dense(features=self.embed_dim)(h)        
        
        # 4. Add the MLP changes back to the main data highway
        x = x + mlp_out
        
        return x

