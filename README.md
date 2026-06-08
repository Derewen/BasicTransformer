# Basic Arithmetic Transformer from Scratch

A toy ~1.2M parameter Decoder-only Transformer built from scratch using Python, JAX, Flax, and Optax. The model is trained to solve simple arithmetic additions (e.g., predicting the answer string `168` from the input prompt `123+45=`) using character-level tokenization.

---

## 1. File Structure

* `embedding_layer.py`: Combines token character embeddings and learned absolute positional embeddings.
* `attention.py`: Causal Multi-Head Self-Attention layer with Query, Key, Value projections and masking.
* `transformer_block.py`: Integrates attention, multi-layer perceptron (MLP), normalization, and residual skip connections.
* `model.py`: Master model class stacking 6 transformer blocks (Depth = 6) and the final output head.
* `main.py`: Main script to generate random additions, calculate masked cross-entropy loss, and execute the compiled GPU training loop.
* `inference.py`: Interactive console script to load saved weights and prompt the model in real-time.
* `eval.py`: Automated batch evaluation script that runs 100 random additions and prints exact match accuracy.
* `docs/`:
  * `basic_overview.md`: Simple conceptual guide on how the model functions.
  * `detailed_technical_implementation.md`: Mathematical and systems-level blueprint.

---

### Important GPU/CUDA Constraint (Windows Users)
> [!WARNING]
> **JAX does not support GPU/CUDA acceleration natively on Windows.** Running natively on Windows forces JAX to run in CPU-only mode. To utilize your NVIDIA GPU (like the RTX 4090) for high-speed training, **you must execute the code inside Windows Subsystem for Linux (WSL2)**.

---

## 2. Installation & Requirements

### A. GPU Execution (Recommended - via WSL2 Ubuntu)
1. Open your WSL2 terminal (Ubuntu).
2. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your-repo-url>
   cd BasicTransformer
   ```
3. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Install JAX with CUDA 12 support (runs directly on your GPU through WSL):
   ```bash
   pip install -U "jax[cuda12]"
   ```

### B. CPU Execution (Fallback - Native Windows)
1. Open your Windows terminal and navigate to the project directory.
2. Install requirements (runs on CPU only):
   ```bash
   pip install -r requirements.txt
   ```


---

## 3. Usage

### A. Training the Model
To train the model from scratch, execute:
```bash
python main.py
```
This runs the training loop for 30,000 steps (configurable in the code), logs the progress, and saves the trained weights to `model.safetensors`.

### B. Interactive Inference Console
To prompt your model interactively (without retraining):
```bash
python inference.py
```
This launches a CLI loop where you can input math equations (e.g., `123+45=`) and view the model's generated answer.

### C. Run Accuracy Evaluations
To check the exact match accuracy of the saved weights on 100 random math questions:
```bash
python eval.py
```
