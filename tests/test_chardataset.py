import pathlib
import importlib.util
import torch


# Load CharDataset module directly from src/CharDataset.py so test doesn't
# rely on package import mechanics.
src_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "CharDataset.py"
spec = importlib.util.spec_from_file_location("chardataset", str(src_path))
chardataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chardataset)
CharDataset = chardataset.CharDataset


class DummyConfig:
    pass


def test_chardataset_basic():
    cfg = DummyConfig()
    cfg.N = 4  # context length

    text = "abcdefghijklmnopqrstuvwxyz"
    ds = CharDataset(cfg, text)

    # vocab size should equal number of unique chars
    assert ds.get_vocab_size() == len(set(text))

    # length: number of valid start positions for chunk of length N+1
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

    # optional helpers
    if hasattr(ds, "encode") and hasattr(ds, "decode"):
        enc = ds.encode("abcd")
        assert ds.decode(enc) == "abcd"
