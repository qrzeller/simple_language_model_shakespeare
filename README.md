# Miniproject 2: Small Language Model (Shakespeare)

This project can be found in the Github repository : [link](https://github.com/qrzeller/simple_language_model_shakespeare)

Clone with : https://github.com/qrzeller/simple_language_model_shakespeare.git
__main__ branch is for mps devices, __cuda__ branch is not the official one.

This project involves the manual implementation of a Transformer-based language model from scratch (using PyTorch for tensors and autograd, but implementing the layers manually).

This project is part of the Deep Learning course by Francois Fleuret at UNIGE.

## Project Requirements & Implementation Status

### Core Architecture (Mandatory)
- [x] **Transformer Decoder-Only Structure**: Implemented in `src/Transformer_Decoder.py`.
- [x] **Input Token Embeddings**: Implemented in `src/Transformer_Decoder.py`.
- [x] **Positional Encodings**: Manually Implemented in `src/positional_encoding.py` (Sinusoidal/Fourier features).
- [x] **Causal Multi-Head Self-Attention**: Manually implemented in `src/MultiHeadAttention.py`.
- [x] **Feed-Forward Neural Networks**: Implemented in `src/DecoderBlock.py`.
- [x] **Residual Connections & Layer Normalization**: Implemented in `src/DecoderBlock.py`.

### Training (Mandatory)
- [x] **Character-level Tokenization**: Implemented in `src/CharDataset.py`.
- [x] **Sliding Window / Context Block**: Handled in `src/CharDataset.py`.
- [x] **Causal Masking**: Implemented in `src/MultiHeadAttention.py`.
- [x] **Adam Optimizer**: Used in `src/main.py`.
- [x] **Cross-Entropy Loss**: Used in `src/main.py`.

### Evaluation & Inference (Mandatory)
- [x] **Cross-Entropy Loss Monitoring**: Tracked per epoch in `src/main.py`.
- [x] **Text Generation**: Implemented `complete_text_generation` in `src/main.py`.

### Optional Requirements
- [x] **Learning Rate Decay**: Implemented in `src/main.py` (supports Cosine, Step, Plateau).
- [x] **Gradient Clipping**: Implemented in `src/main.py` (configurable via `grad_clip`).
- [x] **Perplexity Metric**: Implemented and tracked in `src/main.py`.

## Usage

### Training

To train the model, run the `src/main.py` script:

```bash
python3 src/main.py
```

This script will:
1.  Load the hyperparameters from `dataset/hyperparameters.conf`.
2.  Train the model, saving checkpoints to `checkpoints/`.
3.  Evaluate on validation and test sets.
4.  Generate sample text.
5.  Plot training metrics to `plots/training_metrics.png`.

### Configuration

Hyperparameters are defined in `dataset/hyperparameters.conf`.

## Project Structure

```
.
├── README.md
├── miniproject2_language_model.ipynb  # Original project description notebook
├── dataset/
│   ├── hyperparameters.conf           # Configuration file
│   └── input.txt                      # Shakespeare dataset
├── plots/                             # Training metrics plots
├── checkpoints/                       # Saved model checkpoints
├── src/
│   ├── CharDataset.py                 # Dataset class
│   ├── Config.py                      # Configuration class
│   ├── DecoderBlock.py                # Transformer decoder block
│   ├── MultiHeadAttention.py          # Multi-head attention mechanism
│   ├── Transformer_Decoder.py         # Main Transformer model
│   ├── positional_encoding.py         # Positional encoding
│   ├── plotting.py                    # Plotting utilities
│   └── main.py                        # Main training script
└── tests/                             # Unit tests
```

> # Example of generated text:
> O God, O God! the king is not so far off,
> That I may be so far off that word than the world.
>
> QUEEN MARGARET:
> Then be thy father, and then the world with thy brother's love,
> And then the sea of the death of the wo