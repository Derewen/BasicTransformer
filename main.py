# Imports
from model import SimpleTransformer
from flax.training import train_state
from numpy import random
import numpy as np
import jax
import jax.numpy as jnp
import optax


# Characters
vocab: list[str]= [ '<pad>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '<eos>' ]
char_to_id: dict[str, int] = { char_key : index_value for (index_value, char_key) in enumerate(vocab) }
id_to_char: dict[int, str] = { index_key : char_value for (index_key, char_value) in enumerate(vocab) }

# Char Constants
PAD_ID: int = char_to_id['<pad>']
EOS_ID: int = char_to_id['<eos>']

# Settings
SEQUENCE_LENGTH: int = 16 # Power of 2 is optimal for parallel processing
VOCAB_SIZE: int = len(vocab) # Size of our vocabulary (14 distinct tokens)
EMBEDDING_DIM: int = 128 # Dimension of the embedding vectors

# Functions

## Encoding and Decoding
def encode(text: str, sequence_length: int) -> list[int]:
    ids = [ char_to_id[char] for char in text ]
    ids += [ EOS_ID ]

    if len(ids) > sequence_length:
        raise ValueError(f"Token sequence too long: {len(ids)} > {sequence_length}")

    while (len(ids) < sequence_length):
        ids.append(PAD_ID)

    return ids

def decode(ids: list[int]) -> str:
    chars = [ id_to_char[id] for id in ids ]
    text = "".join(chars) # Reconstruction of string
    text = text.split(id_to_char[EOS_ID])[0].replace(id_to_char[PAD_ID], '') # Cut EOS and Padding
    
    return text

## Data Pipelines

def get_batch(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    batch_encoded: list[ int, list[int] ] = []

    for i in range(batch_size):
        # 1. Generate unique random addition inputs (0 to 999 inclusive)
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)

        # 2. Format and encode
        text: str = f"{a}+{b}={a+b}"
        encoded = encode(text, SEQUENCE_LENGTH)
        batch_encoded.append(encoded)

    # 3. Stack into a contiguous 2D array of shape [B, L]
    array: np.ndarray = np.array(batch_encoded, dtype=np.int32)

    # 4. Data Plumbing: Shift-slice into X and Y tensors of shape [B, L-1]
    x_batch: np.ndarray = array[:, :-1]
    y_batch: np.ndarray = array[:, 1:]

    return x_batch, y_batch

# Loss Function (Grading Engine)
def loss_fn(params, x_batch, y_batch):
    # 1. Run the model forward pass to get raw character scores (logits)
    logits = model.apply({'params': params}, x_batch)
    
    # 2. Compare predicted scores against target IDs (y_batch) using Cross-Entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch)
    
    # 3. Create a mask to ignore padding tokens (ignore target ID 0)
    mask = (y_batch != PAD_ID)
    loss = loss * mask
    
    # 4. Return the average loss of only the active (non-padded) tokens
    return jnp.sum(loss) / jnp.sum(mask)

# Train Step (The execution loop iteration - compiled for GPU speed)
@jax.jit
def train_step(state, x_batch, y_batch):
    # 1. Compute both the loss value and the gradients (how to adjust weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x_batch, y_batch)
    
    # 2. Apply the gradients using our optimizer to update the parameters
    new_state = state.apply_gradients(grads=grads)
    
    # 3. Return the updated train state and the loss value
    return new_state, loss


# Generate/Evaluation Function (Predicts the answer token-by-token)
def generate(params, prompt: str) -> str:
    # 1. Encode prompt string to IDs
    input_ids = encode(prompt, SEQUENCE_LENGTH)
    
    # 2. Strip padding for autoregressive generation
    active_ids = [token_id for token_id in input_ids if token_id != PAD_ID]
    if active_ids[-1] == EOS_ID:
        active_ids = active_ids[:-1]
        
    # 3. Autoregressive Loop (predict character by character)
    for _ in range(8):  # Max 8 steps is enough for a 4-digit answer + EOS
        # Pad our current active list to exactly SEQUENCE_LENGTH
        padded_ids = active_ids + [PAD_ID] * (SEQUENCE_LENGTH - len(active_ids))
        input_tensor = jnp.array([padded_ids], dtype=jnp.int32)
        
        # Slice to SEQUENCE_LENGTH - 1 to match model shape [1, 15]
        model_input = input_tensor[:, :-1]
        
        # Get predictions
        logits = model.apply({'params': params}, model_input)
        
        # Get the predicted character with the highest score at the last active position
        last_index = len(active_ids) - 1
        next_token_id = jnp.argmax(logits[0, last_index])
        next_token_id = int(next_token_id)
        
        # Stop if model outputs End of Sequence
        if next_token_id == EOS_ID:
            break
            
        active_ids.append(next_token_id)
        
        # Stop if we hit safety limit
        if len(active_ids) >= SEQUENCE_LENGTH:
            break
            
    return decode(active_ids)


# Test Functions and Training Loop
if __name__ == "__main__":
    # Generate a batch of 2 samples
    x, y = get_batch(2)
    print("X Batch Shape:", x.shape) # Expect: (2, 15)
    print("Y Batch Shape:", y.shape) # Expect: (2, 15)
    
    print("\nSample 0 Input X:", decode(x[0].tolist()))
    print("Sample 0 Target Y:", decode(y[0].tolist()))

    key = jax.random.PRNGKey(0)
    
    # 1. Initialize the complete SimpleTransformer Model
    model = SimpleTransformer(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, num_heads=4, num_layers=6, max_len=SEQUENCE_LENGTH)
    
    # 2. Initialize parameters using our input x (shape: [2, 15])
    params = model.init(key, x)['params']
    
    # 3. Define the Adam optimizer (handles how we update the weights based on gradients)
    tx = optax.adam(learning_rate=1e-3)
    
    # 4. Create the JAX TrainState (our unified weights container)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # 5. Run the training loop for 30000 steps (for ~100% accuracy)
    print("\nStarting training loop (30,000 steps)...")
    for step in range(30001):
        # Fetch a fresh batch of 128 random math problems
        x_batch, y_batch = get_batch(128)
        
        # Run the compiled train step on the GPU
        state, loss = train_step(state, x_batch, y_batch)
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            print(f"Step {step:5d} | Loss: {loss:.4f}")

            
    print("Training complete!")

    # Save parameters to disk using Safetensors
    import flax
    from safetensors.flax import save_file
    flat_params = flax.traverse_util.flatten_dict(state.params, sep=".")
    save_file(flat_params, "model.safetensors")
    print("Weights successfully saved to model.safetensors!")

    # 6. Evaluate the model on new math problems

    print("\nEvaluating model on new problems:")
    test_queries = ["123+45=", "500+250=", "1+1=", "999+1="]
    for query in test_queries:
        prediction = generate(state.params, query)
        print(f"Prompt: {query:<10} | Model Prediction: {prediction}")