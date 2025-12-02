from dataset.dataset import Synth1Dataset
from model.model import ClapPreTrainModel
from model.loss import MSELoss, CosineSimilarityLoss
from model.trainer import Trainer
from config import DEVICE
import argparse
import tqdm, logging
from datetime import datetime
from utils.upload_model import upload_model_to_hf

def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if args.resume_from_checkpoint:
        date = args.resume_from_checkpoint.split('/')[-2]
    else: 
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = Synth1Dataset(logger=logger, embed_dim=args.embed_dim)
    model = ClapPreTrainModel(embed_dim=args.embed_dim)
    loss_fn = MSELoss() if args.loss == "MSELoss" else CosineSimilarityLoss()
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_fn=loss_fn,
        device=DEVICE,
        checkpoint_path=f'checkpoints/{date}',
        early_stopping_patience=args.es_patience,
        logger=logger
    )
    trainer.train(num_epochs=args.num_epochs, resume_from_checkpoint=args.resume_from_checkpoint)
    if args.upload_to_hf:
        upload_model_to_hf(trainer.model, f'checkpoints/{date}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLAP PreTrain Model")
    parser.add_argument("--loss", type=str, default="MSELoss", choices=["MSELoss", "CosineSimilarityLoss"], help="Loss function to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--es_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--upload_to_hf", type=bool, default=False, help="Upload to Hugging Face")
    args = parser.parse_args()
    main(args)
