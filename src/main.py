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

# Enable cuDNN autotuner for faster operations on H100
torch.backends.cudnn.benchmark = True

def plot_loss(losses=None):
    
    # plot a pretty graph of the training loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    # create directory if not exists
    import os
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    

    # save figure

    plt.savefig('./plots/training_loss.png')

    # show the plot
    plt.show()


def plot_metrics():
    pass  # Plotting logic to be implemented
def complete_text_generation(model, Prompt: str = "O God, O God!", max_length: int = 200, vocab: list[str] = None):

    # We need to do inference with the trained model
    model.eval()
    generated = [ch for ch in Prompt]
    # we could use the method in the CharDataset to encode the prompt, but it's meant for datasets
    input_ids = torch.tensor([[vocab.index(ch) for ch in generated]], dtype=torch.long).to(next(model.parameters()).device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Only feed the model the last `block_size` tokens to avoid exceeding
            # the positional encoding maximum length (model.config.N).
            # We cannot actually give mroe context, the model is not robust to that.
            context = input_ids[:, -model.config.N:]
            outputs = model(context)
            # get the logits for the last token in the sequence, it's not softmaxed yet
            next_token_logits = outputs[0, -1, :]
            # get the token with highest logit
            # unsqueeze twice  because we need (1, 1) shape to concat
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            # append to the input ids
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            # append to the generated string
            generated.append(vocab[next_token_id.item()])
    
    print("Generated text:")
    print("".join(generated))
    
    
    


# inspired from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_epoch(index_epoch, model, training_loader, optimizer, criterion, config : Config, scaler=None):
    running_loss = 0.0
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulation_counter = 0

    for i, data in enumerate(training_loader):
        # input + gt pairs
        inputs, labels = data
        # Move data to the appropriate device with non_blocking for async transfer
        inputs, labels = inputs.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)

        # forward + backward + optimize with mixed precision if enabled
        if scaler is not None:
            # Mixed precision training reduces memory and speeds up H100
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(inputs)
                # outputs: (batch_size, seq_len, vocab_size), labels: (batch_size, seq_len)
                outputs_flat = outputs.view(-1, outputs.size(-1)) # .view(-1, C) is changing (B, S, C) to (B*S, C)
                labels_flat = labels.view(-1) # .view(-1) is changing (B, S) to (B*S,), B*S = total number of tokens in the batch
                loss = criterion(outputs_flat, labels_flat) # cross entropy expects (N, C) and (N,)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(inputs)
            # outputs: (batch_size, seq_len, vocab_size), labels: (batch_size, seq_len)
            outputs_flat = outputs.view(-1, outputs.size(-1)) # .view(-1, C) is changing (B, S, C) to (B*S, C)
            labels_flat = labels.view(-1) # .view(-1) is changing (B, S) to (B*S,), B*S = total number of tokens in the batch
            loss = criterion(outputs_flat, labels_flat) # cross entropy expects (N, C) and (N,)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
        
        accumulation_counter += 1
        
        # Perform optimizer step and gradient reset only after accumulation_steps batches
        if accumulation_counter == accumulation_steps or i == len(training_loader) - 1:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
            
            optimizer.zero_grad()
            accumulation_counter = 0

        # print statistics (use unscaled loss for display)
        running_loss += loss.item() * accumulation_steps

        # Update tqdm description with loss, easyer than average over minibatches
        training_loader.set_postfix(epoch=index_epoch, loss=running_loss / (i + 1))
    
    # Return average loss for the epoch
    avg_epoch_loss = running_loss / len(training_loader)
    return avg_epoch_loss


def train(config: Config, model, train_dataset, loss_fn=nn.CrossEntropyLoss()):
    num_epochs = config.epochs
    model.train()
    losses = []

    batch_size = config.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Setup mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if getattr(config, 'mixed_precision', False) and config.device == 'cuda' else None
    if scaler:
        print("[INFO] Using mixed precision (FP16) training on CUDA")

    # training logic
    for epoch in range(num_epochs):
        # Optimized DataLoader for H100 with more RAM: higher prefetch, more workers
        num_workers = 12 if config.device == 'cuda' else 0
        pin_memory = config.device == 'cuda'
        prefetch_factor = 4 if num_workers > 0 else 0  # Pre-load more batches into GPU memory
        train_loader = tqdm.tqdm(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        ))
        last_loss = train_epoch(epoch, model, train_loader, optimizer, loss_fn, config, scaler)
        losses.append(last_loss)
    
    plot_loss(losses)

    
def evaluate(config: Config, model, val_dataset, loss_fn=nn.CrossEntropyLoss()):
    num_workers = 12 if config.device == 'cuda' else 0
    pin_memory = config.device == 'cuda'
    prefetch_factor = 4 if num_workers > 0 else 0
    val_loader = tqdm.tqdm(DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor
    ))  prefetch_factor=2 if num_workers > 0 else 0
    ))

    # Set model to evaluation mode
    # Disables dropout, activations, etc.
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Ensure no gradients are computed
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move data to the appropriate device with non_blocking for async transfer
            inputs, labels = inputs.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)

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

    # Return metrics for further use if needed
    return avg_loss, accuracy

    


if __name__ == "__main__":

    # load config file, linux style
    cfg = Config.load_from_file("./dataset/hyperparameters.conf")

    text= ""
    with open(cfg.dataset_path, 'r', encoding='utf-8') as f:
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

    train(config=cfg, model=model, train_dataset=char_dataset_train, loss_fn=nn.CrossEntropyLoss())
    evaluate(config=cfg, model=model, val_dataset=char_dataset_val, loss_fn=nn.CrossEntropyLoss())

    plot_loss()  # Function to plot training loss
    plot_metrics()  # Function to plot evaluation metrics

    complete_text_generation(model, vocab=chars)  # Function to generate text after training

