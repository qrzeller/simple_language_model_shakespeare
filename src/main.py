# set python path to src
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.Config import Config
from src.CharDataset import CharDataset
from src.Transformer_Decoder import TransformerDecoder


import tqdm 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

def plot_loss():
    pass  # Plotting logic to be implemented
def plot_metrics():
    pass  # Plotting logic to be implemented
def complete_text_generation():
    pass  # Text generation logic to be implemented


# inspired from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_epoch(index_epoch, model, training_loader, optimizer, criterion, config : Config):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        # input + gt pairs
        inputs, labels = data
        # Move data to the appropriate device
        inputs, labels = inputs.to(config.device), labels.to(config.device) # necessary ?

        # zero the parameter gradients (except for gradacc))
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # outputs: (batch_size, seq_len, vocab_size), labels: (batch_size, seq_len)
        outputs_flat = outputs.view(-1, outputs.size(-1)) # .view(-1, C) is changing (B, S, C) to (B*S, C)
        labels_flat = labels.view(-1) # .view(-1) is changing (B, S) to (B*S,), B*S = total number of tokens in the batch
        loss = criterion(outputs_flat, labels_flat) # cross entropy expects (N, C) and (N,)
        loss.backward()
        
        # eventually we should gives as parameters the metrics to follow (accuracy, perplexity, etc)
        # (lr schedulers) scheduler.step(val_loss)
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # Update tqdm description with loss, easyer than average over minibatches
        training_loader.set_postfix(loss=running_loss / (i + 1))


def train(config: Config, model, loss_fn=nn.CrossEntropyLoss()):
    num_epochs = 1
    model.train()

    # for llms, cross entropy loss is standard (classification per token)
    #loss = nn.CrossEntropyLoss()

    batch_size = config.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # training logic
    for epoch in range(num_epochs):
        train_loader = tqdm.tqdm(DataLoader(char_dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2))
        train_epoch(epoch, model, train_loader, optimizer, loss_fn, config)
    

def evaluate(config: Config, model, loss_fn=nn.CrossEntropyLoss()):
    val_loader = tqdm.tqdm(DataLoader(char_dataset_val, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2))

    # Set model to evaluation mode
    # Disables dropout, activations, etc.
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Ensure no gradients are computed
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss (flatten sequence dimension)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            loss = loss_fn(outputs_flat, labels_flat)
            total_loss += loss.item() * labels_flat.size(0)  # Accumulate total loss over tokens

            # Compute accuracy over tokens
            _, predicted = torch.max(outputs_flat, dim=1)
            total_correct += (predicted == labels_flat).sum().item()
            total_samples += labels_flat.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # TODO: add metric like perplexity, Rouge, BLEU, etc.
    # I would like to try BLEU score if time permits

    # Return metrics for further use if needed
    return avg_loss, accuracy

    


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

    print(f"Dataset vocab size: {char_dataset_train.get_vocab_size()}, Config vocab size: {cfg.vocab_size}")
     # check if model compatible with token
    assert(cfg.vocab_size == char_dataset_train.get_vocab_size() == len(chars)), "Config vocab size does not match dataset vocab size!"

    model = TransformerDecoder(cfg)
    model.to(cfg.device)

    train(config=cfg, model=model, loss_fn=nn.CrossEntropyLoss())
    evaluate(config=cfg, model=model, loss_fn=nn.CrossEntropyLoss())

    plot_loss()  # Function to plot training loss
    plot_metrics()  # Function to plot evaluation metrics

    complete_text_generation()  # Function to generate text after training

