import os
from .trainer import Trainer
from validation.evaluator import Evaluator
from validation.visualization import plot_loss
from config import VOC_CLASSES, VOC_ANCHORS
import torch
from torchvision.datasets import VOCDetection
from dotenv import load_dotenv
from models.mcunet.mcunet.model_zoo import net_id_list, build_model

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


def train_and_save():
    run_name = f"mbv2net_S5_res160_pkg_lrdecay"
    run_dir = os.path.join("./runs", run_name)

    train_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="train", download=False)
    val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)

    trainer = Trainer(
        train_voc_raw,
        anchors=torch.tensor(VOC_ANCHORS, dtype=torch.float32),
        num_classes=len(VOC_CLASSES), 
        image_size=160, # 128, 160, 192, 224, 256
        grid_num=5, # 4, 5, 6, 7, 8
        epoch_num=160,  # 160 in original paper
        batch_size=32,
        aug=False
    )

    trainer.model_construct(net_id="mbv2-w0.35") # mcunet-in4
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

    
    # import torch
    # import torch.nn as nn

    # def print_shapes(backbone, input_size=(1, 3, 224, 224)):
    #     x = torch.zeros(input_size)
    #     print("input:", x.shape)

    #     # Case A: MCUNet-style backbone with first_conv + blocks
    #     if hasattr(backbone, "first_conv") and hasattr(backbone, "blocks"):
    #         x = backbone.first_conv(x)
    #         print("first_conv:", x.shape)

    #         for i, blk in enumerate(backbone.blocks):
    #             x = blk(x)
    #             print(f"blocks[{i}]:", x.shape)

    #     # Case B: MobileNetV2-style backbone with features (Sequential/ModuleList)
    #     elif hasattr(backbone, "features"):
    #         feats = backbone.features
    #         if isinstance(feats, nn.ModuleList):
    #             feats = nn.Sequential(*feats)

    #         for i, m in enumerate(feats):
    #             x = m(x)
    #             print(f"features[{i}]:", x.shape)

    #     else:
    #         # fallback: iterate all children
    #         for name, m in backbone.named_children():
    #             x = m(x)
    #             print(f"{name}:", x.shape)

    # # Example usage
    # backbone, _, _ = build_model(net_id="mbv2-w0.35", pretrained=True)
    # print(backbone)

    # print_shapes(backbone, input_size=(1, 3, 224, 224))