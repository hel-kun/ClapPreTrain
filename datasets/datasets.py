import os, re
import tqdm, logging
import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from dotenv import load_dotenv

class Synth1Dataset(Dataset):
    def __init__(self, logger: logging.Logger = None, embed_dim: int = 512):
        load_dotenv()
        TOKEN = os.getenv("HF_TOKEN")
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Loading Synth1PresetDataset...")
        self.base_data = load_dataset("hel-kun/Synth1PresetDataset", token=TOKEN, trust_remote_code=True)
        self.base_data = self.base_data.cast_column("audio", Audio(sampling_rate=48000))
        self.embed_dim = embed_dim
        self.dataset = self.preprocess()

    def preprocess(self):
        dataset = {
            'train': [],
            'validation': [],
            'test': []
        }
        for split in ['train', 'validation', 'test']:
            bar = tqdm.tqdm(total=len(self.base_data[split]), desc=f"Preprocessing {split} data")
            for item in self.base_data[split]:
                audio_data = item['audio']['array']
                dataset[split].append({
                    'label': item['label'],
                    'audio': audio_data
                })
                bar.update(1)
            bar.close()
        return dataset
    
    def __len__(self) -> int:
        return len(self.base_data)

    def __getitem__(self, idx: int):
        return self.dataset['train'][idx]
    
    def collate_fn(self, batch):
        audio_batch = [item['audio'] for item in batch]
        texts = [item['label']['text'] for item in batch]
        return {
            'audio': audio_batch,
            'texts': texts
        }