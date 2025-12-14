import matplotlib.pyplot as plt
import os

def plot_metrics(metrics):
    # plot a pretty graph of the training metrics

    # create directory if not exists
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Loss
    ax1.plot(epochs, metrics['train_loss'], label='Train Loss')
    ax1.plot(epochs, metrics['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, metrics['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Perplexity
    ax3.plot(epochs, metrics['train_ppl'], label='Train Perplexity')
    ax3.plot(epochs, metrics['val_ppl'], label='Val Perplexity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('Perplexity over Epochs')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('./plots/training_metrics.png')
    plt.show()
