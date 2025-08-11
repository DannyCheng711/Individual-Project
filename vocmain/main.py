import os
from .trainer import Trainer
from validation.evaluator import Evaluator
from validation.visualization import plot_loss
from config import VOC_CLASSES
import torch
from torchvision.datasets import VOCDetection
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


def train_and_save():
    train_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="train", download=False)
    val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)

    trainer = Trainer(
        train_voc_raw,
        num_classes=len(VOC_CLASSES), 
        image_size=160,
        grid_num=5,
        epoch_num=160,  # 160 in original paper
        batch_size=32,
        aug=True
    )

    trainer.model_construct()
    evaluator = Evaluator(
        val_voc_raw,
        trainer.anchors,
        trainer.num_classes,
        trainer.image_size,
        grid_num=trainer.grid_num,
        batch_size=16
    )

    # Train the model
    model, loss_log = trainer.model_train(
        evaluator=evaluator,
        model_path="./saved_models",
        save_image=False
    )

    # Plot loss after training
    suffix = "aug" if trainer.aug else "noaug"
    plot_loss(loss_log, f"./vocmain/images_voc/loss_plot_{trainer.epoch_num}_{suffix}.png")
    print("Training complete. Loss plot saved.")
    

def evaluate_from_checkpoint(checkpoint_path, pkg=False, epoch=None):
    val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)
    # Use the same settings as training
    trainer = Trainer(
        train_voc_raw=None,
        num_classes=len(VOC_CLASSES),
        image_size=160,
        grid_num=5,
        epoch_num=160, # 160 in original paper
        batch_size=32,
        aug=False
    )
    trainer.model_construct()
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()

    evaluator = Evaluator(
        val_voc_raw,
        trainer.anchors,
        trainer.num_classes,
        trainer.image_size,
        grid_num=trainer.grid_num,
        batch_size=16
    )
    evaluator.evaluate(trainer.model, pkg=pkg, epoch=epoch)


if __name__ == "__main__":
    # --- Training ---
    train_and_save()

    # --- Evaluation ---
    # evaluate_from_checkpoint("./saved_models/model_150_aug_epoch_150.pth", pkg=False, epoch=150)

    pass