import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss: list[float], val_loss: list[float], title: str = "Loss History", xlabel: str = "Epoch", ylabel: str = "Loss") -> None:
  plt.figure(figsize=(10, 5))
  plt.plot(train_loss, label="Train Loss")
  plt.plot(val_loss, label="Validation Loss")
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()