import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import ClapPreTrainModel
from typing import Optional
import logging
import tqdm
import os
from utils.plot import plot_loss
from config import DEVICE
from model.loss import MSELoss, CosineSimilarityLoss

class Trainer():
    def __init__(
        self,
        model: ClapPreTrainModel,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        loss_fn: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer: torch.optim.Optimizer = None,
        checkpoint_path: str = 'checkpoint',
        early_stopping_patience: int =10,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.logger = logger or logging.getLogger(__name__)
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        self.train_dataloader = DataLoader(dataset.dataset['train'], batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.dataset['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.dataset['test'], batch_size=self.batch_size
        , shuffle=False, collate_fn=dataset.collate_fn)

    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None) -> None:
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resumed training from checkpoint: {resume_from_checkpoint} at epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0

            train_bar = tqdm.tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", postfix="loss: N/A")
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                audio_inputs = batch['audio']
                text_inputs = batch['texts']
                text_embed, audio_embed = self.model(text_inputs, audio_inputs)
                loss = self.loss_fn(text_embed, audio_embed)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                train_bar.set_postfix_str(f"loss: {loss.item():.4f}")
                train_bar.update(1)
            train_bar.close()
            self.train_losses.append(epoch_loss/len(self.train_dataloader))

            if self.val_dataloader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} train_loss: {epoch_loss/len(self.train_dataloader):.4f} val_loss: {val_loss:.4f}")
            plot_loss(self.train_losses, self.val_losses, save_path=f"{self.checkpoint_path}/loss.png")
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm.tqdm(total=len(self.val_dataloader), desc=f"Validating", postfix="loss: N/A")
            for batch_idx, batch in enumerate(self.val_dataloader):
                audio_inputs = batch['audio']
                text_inputs = batch['texts']
                text_embed, audio_embed = self.model(text_inputs, audio_inputs)
                loss = self.loss_fn(text_embed, audio_embed)
                total_loss += loss.item()
                val_bar.set_postfix_str(f"loss: {loss.item():.4f}")
                val_bar.update(1)
            val_bar.close()
        return total_loss/len(self.val_dataloader)

    def evaluate(self) -> None:
        self.model.eval()
        mse_total_loss = 0.0
        cosine_similarity_total_loss = 0.0
        # MSEとCosineSimilarityLossで評価
        mse_loss = MSELoss()
        cosine_similarity_loss = CosineSimilarityLoss()
        with torch.no_grad():
            test_bar = tqdm.tqdm(total=len(self.test_dataloader), desc=f"Evaluating", postfix="loss: N/A")
            for batch_idx, batch in enumerate(self.test_dataloader):
                audio_inputs = batch['audio']
                text_inputs = batch['texts']
                text_embed, audio_embed = self.model(text_inputs, audio_inputs)
                mse_loss = mse_loss(text_embed, audio_embed)
                cosine_similarity_loss = cosine_similarity_loss(text_embed, audio_embed)
                mse_total_loss += mse_loss.item()
                cosine_similarity_total_loss += cosine_similarity_loss.item()
                test_bar.set_postfix_str(f"loss: {mse_total_loss.item():.4f} {cosine_similarity_total_loss.item():.4f}")
                test_bar.update(1)
            test_bar.close()
        self.logger.info(f"MSE Loss: {mse_total_loss/len(self.test_dataloader):.4f} Cosine Similarity Loss: {cosine_similarity_total_loss/len(self.test_dataloader):.4f}")
        # textfileに保存
        with open(f"{self.checkpoint_path}/evaluation.txt", "w") as f:
            f.write(f"MSE Loss: {mse_total_loss/len(self.test_dataloader):.4f}\n")
            f.write(f"Cosine Similarity Loss: {cosine_similarity_total_loss/len(self.test_dataloader):.4f}\n")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        path: str = f"{self.checkpoint_path}/checkpoint_epoch_{epoch+1}.pth" if not is_best else f"{self.checkpoint_path}/best_model.pth"

        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        if is_best:
            self.logger.info(f"Saved best model checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch']