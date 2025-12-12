from torch.utils.data import Dataset
import torch

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):


        # trasform string to char, (all unique characters in the data)
        self.chars = list(set(data))
        # eventually sort but slower? or sort by occurence?
        # chars = sorted(chars)

        self.stoi = { ch:i for i,ch in enumerate(self.chars) } # map characters to integer indices
        self.itos = {i:ch for i,ch in enumerate(self.chars)} # string to integer

        self.vocab_size = len(self.chars)

        self.data = [self.stoi[ch] for ch in data] # encode the entire dataset as a list of integers
        

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        
        return torch.tensor(self.data[idx:idx+1]), torch.tensor(self.data[idx+1:idx+2])