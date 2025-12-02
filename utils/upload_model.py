# Upload Model to Hugging Face
from huggingface_hub import HfApi, HfFolder, Repository
import os
from dotenv import load_dotenv

def upload_model_to_hf(model, checkpoint_path):
    load_dotenv()
    best_model_path = os.path.join(checkpoint_path, 'best_model.pth')
    TOKEN = os.getenv("HF_TOKEN")
    api = HfApi(token=TOKEN)
    api.upload_file(
        file_path=best_model_path,
        repo_id=f'hel-kun/clap-pretrain-model',
        commit_message=f'Upload best model checkpoint',
        private=True
    )