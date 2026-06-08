import flax.linen as nn
import jax.numpy as jnp

class EmbeddingLayer(nn.Module):
    vocab_size: int
    embed_dim: int = 128
    max_len: int = 16
    
    @nn.compact
    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        # input: [ batch_size, seq_len ]
        batch_size, seq_len = input.shape
        
        # 1. Token Embeddings lookup table; vocab_size (number of rows - per token), embedding_dim (number of columns - width of each row); 
        token_embed: jnp.ndarray = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(input)

        # 2. Position indices generation: [0, 1, 2, ..., L-1]
        positional_encoding_lookup: jnp.ndarray = jnp.arange(seq_len)

        # 3. Position Embeddings lookup table (learned)
        positional_embed = nn.Embed(num_embeddings=self.max_len, features=self.embed_dim)(positional_encoding_lookup)        
        
        # 4. Sum and return
        return token_embed + positional_embed

# Shape: [2, 3, 4] of type float32
# token_embed = [
#   # Batch 0 ("56+")
#   [
#     [ 0.1,  0.2, -0.1,  0.0],  # Vector representing '5'
#     [-0.5,  0.0,  0.8,  0.2],  # Vector representing '6'
#     [ 0.9, -0.9,  0.1,  0.4]   # Vector representing '+'
#   ],
#   # Batch 1 ("012")
#   [
#     [ 0.0,  0.1,  0.2, -0.3],  # Vector representing '0'
#     [ 0.7, -0.2,  0.0,  0.1],  # Vector representing '1'
#     [-0.1,  0.4, -0.5,  0.9]   # Vector representing '2'
#   ]
# ]

# Shape: [3] of type int32
# positional_encoding_lookup = [0, 1, 2]

# Shape: [3, 4] of type float32
# positional_embed = [
#   [ 0.01,  0.02,  0.03,  0.04],  # Row 0: Vector representing "First Position"
#   [-0.01, -0.02, -0.03, -0.04],  # Row 1: Vector representing "Second Position"
#   [ 0.05, -0.05,  0.05, -0.05]   # Row 2: Vector representing "Third Position"
# ]

# Broadcasted positional_embed shape: [2, 3, 4]
# positional_embed_broadcasted = [
#   # Copied for Batch 0
#   [
#     [ 0.01,  0.02,  0.03,  0.04],
#     [-0.01, -0.02, -0.03, -0.04],
#     [ 0.05, -0.05,  0.05, -0.05]
#   ],
#   # Copied for Batch 1
#   [
#     [ 0.01,  0.02,  0.03,  0.04],
#     [-0.01, -0.02, -0.03, -0.04],
#     [ 0.05, -0.05,  0.05, -0.05]
#   ]
# ]

# Sum of token_embed + positional_embed
# output = [
#   # Batch 0
#   [
#     [ 0.1 + 0.01,  0.2 + 0.02, -0.1 + 0.03,  0.0 + 0.04],  # '5' + Pos 0
#     [-0.5 - 0.01,  0.0 - 0.02,  0.8 - 0.03,  0.2 - 0.04],  # '6' + Pos 1
#     [ 0.9 + 0.05, -0.9 - 0.05,  0.1 + 0.05,  0.4 - 0.05]   # '+' + Pos 2
#   ],
#   # Batch 1
#   [
#     [ 0.0 + 0.01,  0.1 + 0.02,  0.2 + 0.03, -0.3 + 0.04],  # '0' + Pos 0
#     [ 0.7 - 0.01, -0.2 - 0.02,  0.0 - 0.03,  0.1 - 0.04],  # '1' + Pos 1
#     [-0.1 + 0.05,  0.4 - 0.05, -0.5 + 0.05,  0.9 - 0.05]   # '2' + Pos 2
#   ]
# ]