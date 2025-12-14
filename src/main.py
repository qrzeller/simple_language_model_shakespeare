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
        loss = criterion(outputs, labels) # cross entropy, no softmax needed
        loss.backward()
        
        # eventually we should gives as parameters the metrics to follow (accuracy, perplexity, etc)
        # (lr schedulers) scheduler.step(val_loss)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{index_epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0 #calculate loss average only over last printed interval



def train(config: Config, model, loss_fn=nn.CrossEntropyLoss()):
    num_epochs = 10
    model.train()

    # for llms, cross entropy loss is standard (classification per token)
    #loss = nn.CrossEntropyLoss()

    batch_size = config.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # training logic
    for epoch in range(num_epochs):
        train_loader = tqdm.tqdm(DataLoader(char_dataset_train, batch_size=batch_size, shuffle=True))
        train_epoch(epoch, model, train_loader, optimizer, loss_fn, config)
    

def evaluate(config: Config, loss_fn=nn.CrossEntropyLoss()):
    val_loader = tqdm.tqdm(DataLoader(char_dataset_val, batch_size=config.batch_size, shuffle=False))

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

            # Compute loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Accumulate total loss

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item() # .item() is used to get a Python number from a tensor
            total_samples += labels.size(0)

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


    print(f"Dataset vocab size: {char_dataset.get_vocab_size()}, Config vocab size: {cfg.vocab_size}")
     # check if model compatible with token

    model = TransformerDecoder(cfg)

    train(config=cfg, model=model, loss_fn=nn.CrossEntropyLoss())
    evaluate(config=cfg, model=model, loss_fn=nn.CrossEntropyLoss())

    plot_loss()  # Function to plot training loss
    plot_metrics()  # Function to plot evaluation metrics

    complete_text_generation()  # Function to generate text after training

