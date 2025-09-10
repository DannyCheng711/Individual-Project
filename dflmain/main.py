import os
import torch 
import json 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.voc.vocdataset import CustomYoloDataset
from models.dethead.yolodet import Yolov2Loss
from config import VOC_CLASSES, VOC_CLASS_TO_IDX, CO3D_ANCHORS
from dotenv import load_dotenv
from PIL import Image
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from utils.config_utils import fix_state_dict_keys

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")
IMAGE_SIZE = 160
S = 5

def freeze_backbone(model):
    for p in model.first_conv.parameters():
        p.requires_grad = False
    for p in model.blocks.parameters():
        p.requires_grad = False

# sdA: the first model's state dict
# NA: the num of samples or occlusion rate
# sdB: the second model's state dict
# NB: the num of samples or occlusion rate 
@torch.no_grad()
def weighted_avg_heads(sdA, NA, sdB, NB):
    out = {}
    denom = float(NA + NB) if (NA + NB) > 0 else 1.0
    for k in sdA.keys():
        out[k] = (sdA[k].float() * NA + sdB[k].float() * NB) / denom
    return out

@torch.no_grad()
def load_head_from_sd(model, sd):
    """Load detection head weight from state dict"""
    missing, unexpected = model.classifier.layer1.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys when loading head: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading head: {unexpected}")


def train_local(model, data_loader, optimizer, device, local_epochs, scheduler=None, base_lr=1e-3, W=5):
    """
    Train detection head locally for 'local_epochs' epochs
    Returns: ttl_samples, avg_loss
    """
    model.train()
    criterion = Yolov2Loss(
        num_classes=len(VOC_CLASSES), 
        anchors=torch.tensor(CO3D_ANCHORS, dtype=torch.float32, device=device)).to(device)

    ttl_samples = 0
    ttl_loss = 0.0
    ttl_batches = 0

    for epoch in range(local_epochs):
        for i, (imgs, targets) in enumerate(data_loader):
        
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(imgs)
            loss_dict = criterion(preds, targets, imgs=imgs)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step() 

            # for whole experiment 
            ttl_batches += 1
            ttl_loss += float(loss)
            ttl_samples += imgs.size(0)

        # Learning rate scheduling
        # warmup → cosine step
        if scheduler is not None:
            if epoch < W:
                warm_frac = (epoch + 1) / W
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * warm_frac
            else:
                scheduler.step()
                
    avg_loss = ttl_loss / max(ttl_batches, 1)
    
    return ttl_samples, avg_loss


# store for logs
# base_lr = 1e-3
# weight_decay = 1e-4
# warmup_epochs = 5
# eta_min = base_lr * 0.05
def fedavg_two_nodes(
        modelA, modelB, train_ds_A, train_ds_B,
        device, rounds=3, batch_size=4, lr=1e-4, local_epochs_per_round=5, 
        weight_decay=1e-4, base_lr=1e-3, warmup_epochs=5
):
    """
    modelA, modelB: identical architectures (separate instances)
    train_ds_*: PyTorch datasets (or wrap list-like into Dataset)
    val_loader: shared validation loader
    """
    # DataLoader preparation
    loaderA = DataLoader(train_ds_A, batch_size=batch_size, shuffle=True)
    loaderB = DataLoader(train_ds_B, batch_size=batch_size, shuffle=True)

    # Move to device and freeze backbone
    modelA.to(device)
    modelB.to(device)

    optA = torch.optim.AdamW(modelA.parameters(), lr=lr, weight_decay=weight_decay)
    optB = torch.optim.AdamW(modelB.parameters(), lr=lr, weight_decay=weight_decay)

    rounds_list = []
    loss_A_list = []
    loss_B_list = []
    avg_loss_list = []

    print(f"Starting FedAvg training for {rounds} rounds...")

    for r in range(1, rounds + 1):
        print(f"\n=== Round {r}/{rounds} ===")
        print("Training Node A...")
        NA, lossA = train_local(
            modelA, loaderA, optA, device, 
            local_epochs=local_epochs_per_round,
            scheduler=None, base_lr=base_lr, W=warmup_epochs
        )
        
        print("Training Node B...")
        NB, lossB = train_local(
            modelB, loaderB, optB, device,
            local_epochs=local_epochs_per_round,
            scheduler=None, base_lr=base_lr, W=warmup_epochs
        )

        print("Performing FedAvg aggregation...")
        sdA = {k: v.detach().cpu().clone() for k, v in modelA.state_dict().items()}
        sdB = {k: v.detach().cpu().clone() for k, v in modelB.state_dict().items()}

        sdAgg = weighted_avg_heads(sdA, NA, sdB, NB)

        modelA.load_state_dict(sdAgg)
        modelB.load_state_dict(sdAgg)

        # Track losses
        avg_loss = (lossA + lossB) / 2
        rounds_list.append(r)
        loss_A_list.append(lossA)
        loss_B_list.append(lossB)
        avg_loss_list.append(avg_loss)
        
        print(f"Samples A/B: {NA}/{NB} | Loss A/B: {lossA:.4f}/{lossB:.4f} | Avg Loss: {avg_loss:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_list, loss_A_list, 'b-o', label='Node A Loss', linewidth=2)
    plt.plot(rounds_list, loss_B_list, 'r-s', label='Node B Loss', linewidth=2)
    plt.plot(rounds_list, avg_loss_list, 'g-^', label='Average Loss', linewidth=2)
    plt.xlabel('FedAvg Round')
    plt.ylabel('Loss')
    plt.title('FedAvg Training Loss vs Round')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./dflmain/fedavg_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Return simplified logs
    logs = {
        'rounds': rounds_list,
        'loss_A': loss_A_list,
        'loss_B': loss_B_list,
        'avg_loss': avg_loss_list
    }
    
    return logs



if __name__ == "__main__":

    eval_class = "car"
    base_path = "./dataset/occlusion/co3d/train_car"

    with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
        manifest = json.load(f)

    with open("./quanmain/model_config/mcunetYolo_config.json", "r") as f:
        mcunetYolo_config = json.load(f)

    checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_lrdecay/best.pth", map_location='cpu')
    print(" Converting PyTorch keys to PyTorch NAS keys...")

    fixed_state_dict = fix_state_dict_keys(checkpoint['model'])

    # Create model A and model B
    modelA = ProxylessNASNets.build_from_config(mcunetYolo_config)
    modelA.load_state_dict(fixed_state_dict)

    modelB = ProxylessNASNets.build_from_config(mcunetYolo_config)
    modelB.load_state_dict(fixed_state_dict)

    all_images1, all_images2, all_gt1, all_gt2 = [], [], [], []

    # Process pairs of images # 171 seqname in training data
    for i in range(0, len(manifest) - 1, 2): # Process pairs 
        item1 = manifest[i]
        item2 = manifest[i + 1]

        filename1 = item1["image"]
        filename2 = item2["image"]

        if os.path.exists(filename1) and os.path.exists(filename2):
            image1 = Image.open(filename1).convert('RGB')
            image2 = Image.open(filename2).convert('RGB')
            all_images1.append(image1)
            all_images2.append(image2)
            gt1 = item1['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            gt2 = item2['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            all_gt1.append([gt1]) # one image contains one ground truth
            all_gt2.append([gt2])

    print(f"Processing {len(all_images1)} image pairs")

    dataset1 = CustomYoloDataset(all_images1, all_gt1, image_size=IMAGE_SIZE, S=S, 
                      anchors=torch.tensor(CO3D_ANCHORS, dtype=torch.float32, device=DEVICE), num_classes=len(VOC_CLASSES), aug=False)

    dataset2 = CustomYoloDataset(all_images2, all_gt2, image_size=IMAGE_SIZE, S=S, 
                      anchors=torch.tensor(CO3D_ANCHORS, dtype=torch.float32, device=DEVICE), num_classes=len(VOC_CLASSES), aug=False)

    
    val_loader = DataLoader(dataset1, batch_size=4, shuffle=False)
    
    # # Run FedAvg
    logs = fedavg_two_nodes(
        modelA, modelB, dataset1, dataset2, device=DEVICE, 
        rounds=40, batch_size=16, lr=1e-4, local_epochs_per_round=5
    )

    # Save logs
    with open('./dflmain/fedavg_logs.json', 'w') as f:
        json.dump(logs, f, indent=2)
    
    print("FedAvg training completed!")
