import matplotlib.pyplot as plt

def plot_loss(
    train_loss: list[float],
    val_loss: list[float],
    title: str = "Loss History",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: str = "loss.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)