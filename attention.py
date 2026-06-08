from flax import traceback_util
import flax.linen as nn
import jax
import jax.numpy as jnp

class CausalSelfAttention(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4  # Must divide embed_dim cleanly (128 / 4 = 32 features per head)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: [ batch_size, seq_len, embed_dim ]
        batch_size, seq_len, embed_dim = x.shape
        head_dim = self.embed_dim // self.num_heads
        
        # 1. Project inputs: Q (what to search), K (indexing metadata), V (actual content payload)
        q = nn.Dense(features=self.embed_dim, name="query")(x)
        k = nn.Dense(features=self.embed_dim, name="key")(x)
        v = nn.Dense(features=self.embed_dim, name="value")(x)
        
        # 2. Split the 128 coordinates into 4 independent channels (heads) of 32 coordinates each
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # 3. Compare Queries and Keys to create a match-score grid (relationship lookup table)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        
        # 4. Hide future tokens by setting their match-scores to negative infinity (no cheating!)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        scores = jnp.where(causal_mask == 1, scores, -1e9)
 
        # 5. Convert raw match-scores into clean percentages that sum to 100% per row
        attention_weights = nn.softmax(scores, axis=-1)
 
        # 6. Grab and blend the content (Values) based on the attention percentages
        out = jnp.matmul(attention_weights, v)
 
        # 7. Glue the 4 channels back into a 128-dimensional vector, and mix the results
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = nn.Dense(features=self.embed_dim, name="output")(out)
        
        return out
