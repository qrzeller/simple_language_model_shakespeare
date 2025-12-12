from Config import Config
from CharDataset import CharDataset
from Transformer_Decoder import TransformerDecoder


def plot_loss():
    pass  # Plotting logic to be implemented
def plot_metrics():
    pass  # Plotting logic to be implemented
def complete_text_generation():
    pass  # Text generation logic to be implemented



def train():
    pass  # Training logic to be implemented

def evaluate():
    pass  # Evaluation logic to be implemented


if __name__ == "__main__":
    text= ""
    with open('./dataset/input.txt', 'r') as f:
        text = f.read()

    cfg = Config.load_from_file("./dataset/hyperparameters.conf")
    char_dataset = CharDataset(cfg, text)

    print(f"Dataset vocab size: {char_dataset.get_vocab_size()}, Config vocab size: {cfg.vocab_size}")
     # check if model compatible with token

    model = TransformerDecoder(cfg)

    train()
    evaluate()

    plot_loss()  # Function to plot training loss
    plot_metrics()  # Function to plot evaluation metrics

    complete_text_generation()  # Function to generate text after training

    