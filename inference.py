import jax
import jax.numpy as jnp
import flax
from safetensors.flax import load_file
from model import SimpleTransformer
from main import encode, decode, PAD_ID, EOS_ID, SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM

# Local generate function to avoid global scope issues
def generate(model, params, prompt: str) -> str:
    # 1. Encode prompt string to IDs
    input_ids = encode(prompt, SEQUENCE_LENGTH)
    
    # 2. Strip padding for autoregressive generation
    active_ids = [token_id for token_id in input_ids if token_id != PAD_ID]
    if active_ids[-1] == EOS_ID:
        active_ids = active_ids[:-1]
        
    # 3. Autoregressive Loop (predict character by character)
    for _ in range(8):  # Max 8 steps is enough for a 4-digit answer + EOS
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

def main():
    print("Loading model and weights...")
    
    # 1. Initialize the model structure (must match training parameters)
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        num_heads=4,
        num_layers=6,
        max_len=SEQUENCE_LENGTH
    )
    
    # 2. Load the flat safetensors weights dictionary from disk
    try:
        flat_params = load_file("model.safetensors")
    except FileNotFoundError:
        print("Error: 'model.safetensors' not found. Please run main.py first to train the model and save the weights.")
        return
        
    # 3. Reconstruct the nested dictionary structure that Flax expects
    params = flax.traverse_util.unflatten_dict(flat_params, sep=".")
    
    # 4. Run interactive console loop
    print("\n========================================================")
    print("  Interactive Arithmetic Transformer Console")
    print("========================================================")
    print("  Prompt Format: A+B= (e.g., '123+45=' or '150+150')")
    print("  Note: Numbers A and B should be integers from 0 to 999.")
    print("        If '=' is omitted, it will be added automatically.")
    print("  Type 'exit' or 'quit' to close.")
    print("========================================================")

    
    while True:
        try:
            prompt = input("\nEnter math prompt (e.g. '150+150='): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not prompt:
            continue
            
        if prompt.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        # Ensure the prompt ends with the equals sign
        if not prompt.endswith("="):
            prompt += "="
            
        # Basic validation: ensure prompt only contains vocabulary characters
        invalid_chars = [char for char in prompt if char not in "0123456789+="]
        if invalid_chars:
            print(f"Error: Invalid characters in prompt: {invalid_chars}")
            print("Only digits, '+' and '=' are allowed.")
            continue
            
        # Parse and validate integer range (0 to 999)
        parse_prompt = prompt.rstrip("=")
        parts = parse_prompt.split("+")
        if len(parts) == 2:
            try:
                val_a = int(parts[0])
                val_b = int(parts[1])
                if val_a > 999 or val_b > 999:
                    print(f"Error: Numbers must be between 0 and 999 (you entered {val_a} and {val_b}).")
                    print("The model was trained exclusively on 3-digit addition problems.")
                    continue
            except ValueError:
                print("Error: Invalid addition format. Use 'A+B=' where A and B are integers.")
                continue
        else:
            print("Error: Invalid addition format. Use 'A+B=' where A and B are integers.")
            continue

            
        # Run inference
        prediction = generate(model, params, prompt)
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
