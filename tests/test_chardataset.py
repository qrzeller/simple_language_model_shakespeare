import sys
import pathlib
import torch

# Make sure repo root is on sys.path so `src` package can be imported
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.Config import Config
from src.CharDataset import CharDataset


def test_chardataset_basic():
    # load the real config file (we assert it contains block_size)
    cfg = Config.load_from_file("dataset/hyperparameters.conf")
    assert hasattr(cfg, "block_size")

    # For test purposes use a small context length so chunks fit in test text
    cfg.N = 4

    text = "abcdefghijklmnopqrstuvwxyz"
    ds = CharDataset(cfg, text)

    # vocab size should equal number of unique chars
    assert ds.get_vocab_size() == 26

    # length: number of valid start positions for chunk of length N+1
    # since all chars are unique in test text, this is len(text) - N
    assert len(ds) == len(text) - cfg.N

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape[0] == cfg.N
    assert y.shape[0] == cfg.N

    # the target is the input shifted by one
    assert torch.equal(x[1:], y[:-1])
    # check actual values
    expected_x0 = torch.tensor([ds.stoi[ch] for ch in "abcd"], dtype=torch.long)
    expected_y0 = torch.tensor([ds.stoi[ch] for ch in "bcde"], dtype=torch.long)
    assert torch.equal(x, expected_x0)
    assert torch.equal(y, expected_y0)
