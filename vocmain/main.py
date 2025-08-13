import os
from .trainer import Trainer
from validation.evaluator import Evaluator
from validation.visualization import plot_loss
from config import VOC_CLASSES, VOC_ANCHORS
import torch
from torchvision.datasets import VOCDetection
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


def train_and_save():
    run_name = f"mcunet_S8_res256_pkg_lrdecay"
    run_dir = os.path.join("./runs", run_name)

    train_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="train", download=False)
    val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)

    trainer = Trainer(
        train_voc_raw,
        anchors=torch.tensor(VOC_ANCHORS, dtype=torch.float32),
        num_classes=len(VOC_CLASSES), 
        image_size=256, # 128, 160, 192, 224, 256
        grid_num=8, # 4, 5, 6, 7, 8
        epoch_num=160,  # 160 in original paper
        batch_size=32,
        aug=False
    )

    trainer.model_construct()
    evaluator = Evaluator(
        val_voc_raw,
        trainer.anchors,
        trainer.num_classes,
        trainer.image_size,
        grid_num=trainer.grid_num,
        batch_size=16, 
        epoch_num=trainer.epoch_num,
        save_dir=run_dir,
        pkg=True
    )

    # Train the model
    model, loss_log = trainer.model_train(
        evaluator=evaluator,
        model_path=run_dir
    )

    # Plot loss after training
    suffix = "aug" if trainer.aug else "noaug"
    plot_loss(loss_log, os.path.join(run_dir, f"loss_plot_{trainer.epoch_num}_{suffix}.png"))
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

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.model.eval()

    evaluator = Evaluator(
        val_voc_raw,
        trainer.anchors,
        trainer.num_classes,
        trainer.image_size,
        grid_num=trainer.grid_num,
        batch_size=16, 
        epoch_num=trainer.epoch_num,
        save_dir=None,
        pkg=True
    )
    evaluator.evaluate(trainer.model, epoch=epoch)


if __name__ == "__main__":
    # --- Training ---
    train_and_save()

    # --- Evaluation ---
    # evaluate_from_checkpoint("./runs/mcunet_S5_res160_pkg/best.pth", pkg=False, epoch=150)

    # pass