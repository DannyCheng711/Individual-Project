from mcunet.mcunet.model_zoo import net_id_list, build_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import os
import torch
from config import DEVICE, VOC_ROOT
from validation import eval_metrics, eval_metrics_pkg
import preprocess
import model
from validation import decode_pred, visualize_predictions

# Setting
num_classes = 20

"""
# COCO
anchors = [[11.579129865030023 / 32 , 23.897646887526435 / 32], 
    [22.5801077336668254 / 32, 52.59153602786403 / 32], 
    [36.01190595457166 / 32, 90.59964938081846 / 32], 
    [56.857930905762814 / 32, 133.10148018691575 / 32], 
    [125.51095889848243/ 32, 134.68714790018956 / 32]]
"""

class PreTrainStage():
    def __init__(self, train_voc_raw, val_voc_raw,  num_classes, image_size, epoch_num, aug):
        
        # anchor is grid unit 
        self.anchors = torch.tensor([
            [8.851960381366869 / 32, 20.71491425034285 / 32], 
            [27.376232242192298 / 32, 56.73302518805578 / 32], 
            [42.88177452824786 / 32, 98.24243329589638 / 32], 
            [67.68032082718717 / 32, 132.7704493338952 / 32], 
            [131.16250016574756 / 32, 137.1476847579408 / 32]
        ], dtype=torch.float32)

        self.num_classes = num_classes # VOC has 20 classes
        self.image_size = image_size # mcunet
        self.epoch_num = epoch_num  #20
        self.train_voc_raw = train_voc_raw
        self.val_voc_raw = val_voc_raw
        self.aug = aug
        self.net = None
        self.loss_fn = None
        self.optimiser = None

    def yolo_collate_fn(self, batch):
        images = []
        targets = []

        for img, target in batch:
            images.append(img)
            targets.append(target)  # keep list of [num_objects_i, 5] tensors

        return torch.stack(images, dim=0).to(DEVICE), torch.stack(targets, dim=0).to(DEVICE) # [B, 3, H, W], [B, S, S, A, 5 + C]

    def save_preds_image(self, img_tensor, pred_tensor, epoch, batch_idx, img_idx, save_dir="./images_voc", threshold = 0.5):
        """
        pred_tensor: Tensor of shape [B, A*(5+C), S, S] 
        """

        os.makedirs(save_dir, exist_ok=True)
        img = transforms.functional.to_pil_image(img_tensor.cpu())
        draw = ImageDraw.Draw(img)

        decoded_boxes = decode_pred(
            pred_tensor.unsqueeze(0), # add batch dimension
            anchors = self.anchors,
            num_classes = self.num_classes,
            image_size = self.image_size,
            conf_thresh = threshold
        )

        # Draw boxes for first image
        for box in decoded_boxes[0]:  # Get first batch element
            xmin, ymin, xmax, ymax, conf, class_id = box
            
            # Draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin - 10), f"Class {int(class_id)}: {conf:.2f}", fill="red")

        filename = f"epoch_{epoch}_batch_{batch_idx}_image_{img_idx}.jpg"
        img.save(os.path.join(save_dir, filename))


    def plot_loss(self, loss_log, save_path):
        plt.figure(figsize=(10, 6))
        for k in loss_log:
            plt.plot(loss_log[k], label = k)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def model_construct(self):
        """
        # Import the backbone function
        from mcunet.mcunet.tinynas.elastic_nn.networks.ofa_proxyless import ProxylessNASNets
        
        # Create backbone
        backbone = ProxylessNASNets.build_from_config({
            'name': 'ProxylessNASNets',
            'bn': {'name': 'BatchNorm2d'},
            'first_conv': {'name': 'ConvLayer', 'kernel_size': 3, 'stride': 2, 'dilation': 1, 'groups': 1, 'bias': False, 'has_shuffle': False, 'use_se': False, 'act_func': 'relu6'},
            'blocks': [
                # ... your backbone config here
            ],
            'feature_mix_layer': None,
            'classifier': None
        })
        """
        print(net_id_list)
        self.anchors = self.anchors.to(DEVICE)
        backbone, _, _ = build_model(net_id="mcunet-in4", pretrained=True)

        # Load model and loss 
        self.net = model.McuYolo(backbone_fn=backbone, num_classes=self.num_classes, num_anchors=len(self.anchors)).to(DEVICE)
        self.loss_fn = model.Yolov2Loss(num_classes=self.num_classes, anchors=self.anchors).to(DEVICE)
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-4)

    def model_train(self, image_size):
        
        # batch_size: number of images (samples) processed together in one forward pass
        # images: a tensor of batch_size images, shape [2, C, H, W]
        # targets: a list of batch_size elements 

        anchors_list = self.anchors.tolist()

        train_dataset = preprocess.YoloVocDataset(
            voc_dataset=self.train_voc_raw, 
            image_size=image_size,
            S=5,  # Grid size
            anchors=anchors_list,
            num_classes=self.num_classes,
            aug = self.aug
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=self.yolo_collate_fn)

        # Training Loop
        loss_log = {'total': [], 'coord': [], 'obj': [], 'class': []} 
        for epoch in range(self.epoch_num):
            self.net.train()
            # print(f"Model is on device: {next(net.parameters()).device}")
            # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
            # i: counts which batch you’re on (starting from 0).
            # targets: a list of batch_size elements
            # step: (data number / batchsize)
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(DEVICE)

                preds = self.net(imgs)

                loss_dict = self.loss_fn(preds, targets, imgs = imgs)
                loss = loss_dict['total']    

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step() 

                 # Logging
                loss_log['total'].append(loss.item())
                loss_log['coord'].append(loss_dict['coord'].item())
                loss_log['obj'].append(loss_dict['obj'].item())
                loss_log['class'].append(loss_dict['class'].item())  # Fixed: Added class loss
        
                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}] Batch {i} "
                        f"Loss: {loss.item():.4f}, "
                        f"Coord: {loss_dict['coord'].item():.4f}, "
                        f"Obj: {loss_dict['obj'].item():.4f}, "
                        f"Class: {loss_dict['class'].item():.4f}"
                    )

                # Save prediction images on last epoch
                if epoch == self.epoch_num - 1 and i % 50 == 0:  # Save less frequently
                    for img_idx in range(min(2, imgs.size(0))):
                        self.save_preds_image(imgs[img_idx], preds[img_idx], epoch, i, img_idx)

            # Validation and checkpoint saving
            if epoch % 10 == 0:
                self.model_val(epoch=epoch)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss_log': loss_log
                }
                os.makedirs("./saved_models", exist_ok=True)
                torch.save(checkpoint, f"./saved_models/yolovoc_150_aug_epoch_{epoch}.pth")

        return self.net, loss_log

    def train_result_save(self, net, loss_log, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.plot_loss(loss_log, save_path = save_path)

        # Save final model
        model_save_path = "./saved_models/yolovoc_final_150_aug.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(net.state_dict(), model_save_path)
        print(f" Model saved to {model_save_path}")
    
    def load_model(self, model_path):
        """
        Load model from checkpoint file
        Args:
            model_path: path to the checkpoint file
        Returns:
            loaded_info: dict with epoch, loss_log info
        """
        self.model_construct()

        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Full checkpoint format
        self.net.load_state_dict(checkpoint['model_state_dict'])
    
        # Optionally load optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimiser is not None:
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get additional info
        epoch = checkpoint.get('epoch', 'unknown')
        loss_log = checkpoint.get('loss_log', None)
    
        print(f"Model loaded from {model_path}")
        print(f"Checkpoint epoch: {epoch}")

        if loss_log:
            print(f"Available loss keys: {list(loss_log.keys())}")
            print(f"Total loss entries: {len(loss_log['total'])}")

        self.net.eval() # ensures correct test-time behavior!
        
        return {
            'epoch': epoch, 
            'loss_log': loss_log, 
            'checkpoint_path': model_path
        }

    def model_val(self, epoch = None, pkg = False):

        anchors_list = self.anchors.tolist()

        val_dataset = preprocess.YoloVocDataset(
            voc_dataset=self.val_voc_raw, 
            image_size=self.image_size,
            S=5,
            anchors=anchors_list,
            num_classes=self.num_classes
        )

        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=self.yolo_collate_fn)

        # Validation
        self.net.eval()

        """
        train_dataset = preprocess.YoloVocDataset(
            voc_dataset=self.train_voc_raw, 
            image_size=self.image_size,
            S=5,  # Grid size
            anchors=anchors_list,
            num_classes=self.num_classes
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=self.yolo_collate_fn)
        """

        visualize_predictions(
            val_loader=val_loader, 
            model=self.net, 
            anchors=self.anchors,  # Pass tensor anchors
            conf_thresh=0.5
        )
        if pkg:
            map_result = eval_metrics_pkg(
                val_loader, self.net, self.anchors, num_classes=20, device=DEVICE)
            print(map_result)
        else:
            eval_metrics(
                val_loader, self.net, self.anchors, epoch, iou_threshold=0.5)


if __name__ == "__main__":
    
    train_voc_raw = VOCDetection(
        root=VOC_ROOT, year="2012", image_set="train", download=False)
    val_voc_raw = VOCDetection(
        root=VOC_ROOT, year="2012", image_set="val", download=False)
      
    trainer = PreTrainStage(train_voc_raw, val_voc_raw, num_classes=20, image_size=160, epoch_num=150, aug = True)
    # trainer.model_construct()
    # net, log = trainer.model_train(image_size=160)
    # trainer.train_result_save(net, log, "./images_voc/loss_plot_150_aug.png")
    record = trainer.load_model("./saved_models/yolovoc_150_aug_epoch_80.pth")
    # torch.save(trainer.net, "./saved_models/yolovoc_150_aug_epoch_80.pkl")
    # print("Model successfully saved as pkl!")
    trainer.model_val(pkg=False, epoch=None)
    