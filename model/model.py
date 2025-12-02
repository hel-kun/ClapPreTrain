import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, AutoProcessor, ClapAudioModelWithProjection

class ClapTextEncoder(nn.Module):
    def __init__(
        self,
        model_name='laion/clap-htsat-unfused',
        embed_dim=512
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ClapTextModelWithProjection.from_pretrained(model_name)

        if self.model.config.projection_dim != embed_dim:
            self.project = nn.Linear(self.model.config.projection_dim, embed_dim)
        else:
            self.project = nn.Identity()

    def forward(self, input_texts):
        inputs = self.processor(
            text=input_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        embeddings = outputs.text_embeds  # (batch_size, projection_dim)
        embeddings = self.project(embeddings) # 次元変換
        embeddings = embeddings.unsqueeze(1)  # seq_len次元を追加(batch_size, 1, dim)
        
        return embeddings
    
class CLAPAudioEncoder(nn.Module):
    def __init__(
        self,
        model_name="laion/clap-htsat-unfused",
        embed_dim=512,
    ):
        super().__init__()
        self.model = ClapAudioModelWithProjection.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.model.config.projection_dim != embed_dim:
            self.project = nn.Linear(self.model.config.projection_dim, embed_dim)
        else:
            self.project = nn.Identity()

        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, input_audio):
        with torch.no_grad():
            inputs = self.processor(
                audio=input_audio,
                sampling_rate=48000,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embeddings = outputs.audio_embeds  # (batch_size, projection_dim)
            embeddings = self.project(embeddings)  # 次元変換
            embeddings = embeddings.unsqueeze(1)  # seq_len次元を追加(batch_size, 1, dim)
        
        return embeddings

class ClapPreTrainModel(nn.Module):
    def __init__(
        self,
        text_model_name='laion/clap-htsat-unfused',
        audio_model_name='laion/clap-htsat-unfused',
        embed_dim=512
    ):
        super().__init__()
        self.text_encoder = ClapTextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim
        )
        self.audio_encoder = CLAPAudioEncoder(
            model_name=audio_model_name,
            embed_dim=embed_dim
        )
        
    def forward(self, input_texts, input_audio):
        text_embeddings = self.text_encoder(input_texts)  # (batch_size, 1, dim)
        audio_embeddings = self.audio_encoder(input_audio)  # (batch_size, 1, dim)
        
        return text_embeddings, audio_embeddings