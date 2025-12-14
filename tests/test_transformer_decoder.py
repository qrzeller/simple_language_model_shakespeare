import sys
import pathlib
import torch

# Make sure repo root is on sys.path so `src` package can be imported
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.Config import Config
from src.Transformer_Decoder import TransformerDecoder


def test_transformer_decoder_basic():
    # Create a minimal config for testing
    cfg = Config()
    cfg.model_dim = 64
    cfg.num_heads = 4
    cfg.num_layers = 2
    cfg.vocab_size = 26  # Small vocab for test
    cfg.max_seq_length = 10
    cfg.N = cfg.max_seq_length  # For causal mask
    cfg.dropout = 0.1  # Dropout rate
    cfg.device = 'cpu'  # Use CPU for tests

    # Instantiate the model
    model = TransformerDecoder(cfg)
    model.eval()  # Set to eval mode to avoid dropout randomness

    # Create dummy input: batch_size=2, seq_len=5
    batch_size = 2
    seq_len = cfg.max_seq_length  # Match the mask size
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(x)

    # Check output shape: (batch_size, seq_len, vocab_size)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), f"Expected {(batch_size, seq_len, cfg.vocab_size)}, got {logits.shape}"

    # Check that logits are not all zeros (model is producing output)
    assert not torch.allclose(logits, torch.zeros_like(logits)), "Model output is all zeros"

    print("TransformerDecoder test passed!")


if __name__ == "__main__":
    test_transformer_decoder_basic()