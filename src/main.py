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
import math

def plot_loss(losses):
    
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


# only works for auto-regressive models
# https://huggingface.co/docs/transformers/perplexity
# Perplexity is defined as the exponentiated average negative log-likelihood of a sequence.
# It's dependent on the token count : Cannot be compared across models
# Equivalent to exp(cross-entropy loss)
# 
def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss))


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
def train_epoch(index_epoch, model, training_loader, optimizer, criterion, config : Config, scheduler=None):
    running_loss = 0.0

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

        # If the scheduler is configured to step every batch, step it here.
        # Note: `ReduceLROnPlateau` expects a metric and should be stepped per epoch.
        if scheduler is not None and getattr(config, 'scheduler_step_per', 'epoch') == 'batch':
            # avoid calling step() for ReduceLROnPlateau here
            if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        # print statistics
        running_loss += loss.item()

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

    # Setup LR scheduler according to config (if any)
    scheduler = None
    sched_name = getattr(config, 'scheduler', 'none').lower()
    # If user requests cosine decay with warmup, create a LambdaLR that
    # linearly increases lr for `warmup_steps` then applies cosine decay
    warmup_steps = getattr(config, 'scheduler_warmup_steps', 0)
    if sched_name == 'cosine' and warmup_steps and warmup_steps > 0:
        # compute total training steps (epochs * steps_per_epoch)
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        total_steps = max(1, config.epochs * steps_per_epoch)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # progress in [0, 1]
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # LambdaLR should be stepped every optimizer step (per-batch)
        config.scheduler_step_per = 'batch'
    elif sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=getattr(config, 'scheduler_step_size', 10), gamma=getattr(config, 'scheduler_gamma', 0.1))
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(config, 'scheduler_T_max', 50), eta_min=0.0)
    elif sched_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=getattr(config, 'scheduler_gamma', 0.1), patience=getattr(config, 'scheduler_patience', 5))
    else:
        scheduler = None

    # training logic
    for epoch in range(num_epochs):
        train_loader = tqdm.tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2))
        last_loss = train_epoch(epoch, model, train_loader, optimizer, loss_fn, config, scheduler=scheduler)
        losses.append(last_loss)

        # Step scheduler once per epoch if configured. For LambdaLR with warmup
        # we configured `scheduler_step_per='batch'` and it will be stepped inside
        # the batch loop. For ReduceLROnPlateau we pass the epoch loss.
        if scheduler is not None and getattr(config, 'scheduler_step_per', 'epoch') == 'epoch':
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(last_loss)
            else:
                scheduler.step()

        # Print learning rate for monitoring
        if scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: lr={lr:.6e}")
    
    return losses

    

def evaluate(config: Config, model, val_dataset, loss_fn=nn.CrossEntropyLoss()):
    val_loader = tqdm.tqdm(DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2))

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

    # Compute perplexity from the averaged cross-entropy loss.
    # Perplexity = exp(average_negative_log_likelihood) = exp(avg_loss)
    perplexity = calculate_perplexity(avg_loss).item()
    

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Perplexity: {perplexity:.4f}")

    # Return metrics for further use if needed
    return avg_loss, accuracy, perplexity

    


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

    losses = train(config=cfg, model=model, train_dataset=char_dataset_train, loss_fn=nn.CrossEntropyLoss())
    evaluate(config=cfg, model=model, val_dataset=char_dataset_val, loss_fn=nn.CrossEntropyLoss())

    complete_text_generation(model, vocab=chars)  # Function to generate text after training
    plot_loss(losses)  # Function to plot training loss
    
    # calculate perpexity since we have a fixed length model


    plot_metrics()  # Function to plot evaluation metrics

    

