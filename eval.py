import jax
import jax.numpy as jnp
import flax
from numpy import random
from safetensors.flax import load_file
from model import SimpleTransformer
from main import encode, decode, PAD_ID, EOS_ID, SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM
from inference import generate

def main():
    print("Loading model and weights for evaluation...")
    
    # 1. Initialize model structure
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        num_heads=4,
        num_layers=6,
        max_len=SEQUENCE_LENGTH
    )
    
    # 2. Load weights
    try:
        flat_params = load_file("model.safetensors")
    except FileNotFoundError:
        print("Error: 'model.safetensors' not found. Please run main.py first to train the model.")
        return
        
    params = flax.traverse_util.unflatten_dict(flat_params, sep=".")
    
    # 3. Generate 100 random math problems
    print("\nGenerating 100 random math problems...")
    num_tests = 100
    exact_matches = 0
    total_numeric_error = 0
    valid_predictions_count = 0
    
    for i in range(num_tests):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        expected_ans = a + b
        
        prompt = f"{a}+{b}="
        prediction = generate(model, params, prompt)
        
        # Parse the predicted answer
        # The returned prediction is of format: "a+b=ans"
        try:
            predicted_ans_str = prediction.split("=")[-1]
            predicted_ans = int(predicted_ans_str)
            
            # Check for exact matches
            if predicted_ans == expected_ans:
                exact_matches += 1
                
            error = abs(predicted_ans - expected_ans)
            total_numeric_error += error
            valid_predictions_count += 1
            
        except ValueError:
            # Model prediction was not a valid integer (e.g. contained symbols or was empty)
            print(f"Test {i+1:3d} | Prompt: {prompt:<10} | Failed to parse output: '{prediction}'")
            total_numeric_error += expected_ans  # Treat total value as error
            
    # 4. Print results
    print("\n==============================================")
    print("  Evaluation Results (100 Random Tests)")
    print("==============================================")
    accuracy = (exact_matches / num_tests) * 100
    print(f"Exact Match Accuracy: {accuracy:.2f}% ({exact_matches}/{num_tests})")
    
    if valid_predictions_count > 0:
        avg_error = total_numeric_error / num_tests
        print(f"Average Numerical Error Distance: {avg_error:.2f}")
    else:
        print("Average Numerical Error Distance: N/A (all predictions failed to parse)")
        
    print("\nNote: A 1.2M model trained for 30,000 steps will typically get ~95-100% exact accuracy.")


if __name__ == "__main__":
    main()
