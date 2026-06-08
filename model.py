import flax.linen as nn
import jax.numpy as jnp
from embedding_layer import EmbeddingLayer
from transformer_block import TransformerBlock

class SimpleTransformer(nn.Module):
    vocab_size: int
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 6  # This defines the Depth (number of blocks stacked)
    max_len: int = 16
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: [ batch_size, seq_len ] (discrete integer token IDs)
        
        # 1. Entry Gate: Convert input token IDs into 128-dimensional coordinate embeddings
        x = EmbeddingLayer(vocab_size=self.vocab_size, embed_dim=self.embed_dim, max_len=self.max_len)(x)
        
        # 2. Sequential Processing (Depth): Pass coordinates through 6 sequential blocks
        for layer_index in range(self.num_layers):
            x = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)(x)
 
        # 3. Final Leveling: Normalize values to keep them stable before output
        x = nn.LayerNorm()(x)
        
        # 4. Output Head: Translate the 128 coordinates back into 14 character scores (logits)
        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
 
        return logits
