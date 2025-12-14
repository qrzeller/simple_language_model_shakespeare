from src.Config import Config
from src.CharDataset import CharDataset
from src.Transformer_Decoder import TransformerDecoder


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

    # load config file, linux style
    cfg = Config.load_from_file("./dataset/hyperparameters.conf")

    text= ""
    with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()


    # create the splits, 90, 5, 5 %
    n = len(text)
    train_data = text[:int(n*0.9)]
    val_data = text[int(n*0.9):int(n*0.95)]
    test_data = text[int(n*0.95):]

    # TODO : we should avoid leakage between the plays


    chars = list(dict.fromkeys(text))
    char_dataset_train = CharDataset(cfg, train_data, vocab=chars)
    char_dataset_val = CharDataset(cfg, val_data, vocab=chars)   
    char_dataset_test = CharDataset(cfg, test_data, vocab=chars)


    print(f"Dataset vocab size: {char_dataset.get_vocab_size()}, Config vocab size: {cfg.vocab_size}")
     # check if model compatible with token

    model = TransformerDecoder(cfg)

    train()
    evaluate()

    plot_loss()  # Function to plot training loss
    plot_metrics()  # Function to plot evaluation metrics

    complete_text_generation()  # Function to generate text after training

    