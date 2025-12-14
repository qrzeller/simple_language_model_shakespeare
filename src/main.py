from src.Config import Config
from src.CharDataset import CharDataset
from src.Transformer_Decoder import TransformerDecoder

import tqdm 

def plot_loss():
    pass  # Plotting logic to be implemented
def plot_metrics():
    pass  # Plotting logic to be implemented
def complete_text_generation():
    pass  # Text generation logic to be implemented


# inspired from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_epoch(index_epoch, model, training_loader, optimizer, criterion):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        # input + gt pairs
        inputs, labels = data
        # zero the parameter gradients (except for gradacc))
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels) # cross entropy, no softmax needed
        loss.backward()
        
        # eventually we should gives as parameters the metrics to follow (accuracy, perplexity, etc)
        # (lr schedulers) scheduler.step(val_loss)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{index_epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0



def train():
    pass

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

    