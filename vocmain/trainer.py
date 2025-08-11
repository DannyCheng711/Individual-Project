import os 
# Third-party
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
# Local application
from config import VOC_ANCHORS, VOC_CLASSES
from dataset.vocdatset import YoloVocDataset
from models.mcunet.mcunet.model_zoo import net_id_list, build_model
from models.dethead.yolodet import McuYolo, Yolov2Loss
from validation.visualization import save_preds_image

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

class Trainer:
    def __init__(self, train_voc_raw, num_classes, image_size, grid_num, epoch_num, batch_size, aug):
        # anchor is grid unit 
        self.anchors = torch.tensor(VOC_ANCHORS, dtype=torch.float32)
        self.num_classes = num_classes # VOC has 20 classes
        self.image_size = image_size # mcunet
        self.epoch_num = epoch_num  
        self.grid_num = grid_num
        self.batch_size = batch_size
        self.train_voc_raw = train_voc_raw
        self.aug = aug
        self.model = None
        self.criterion = None
        self.optimizer = None

    def yolo_collate_fn(self, batch):
        images, targets = [], []
        for img, target in batch:
            images.append(img)
            targets.append(target)  # keep list of [num_objects_i, 5] tensors
        return torch.stack(images, dim=0).to(DEVICE), torch.stack(targets, dim=0).to(DEVICE) # [B, 3, H, W], [B, S, S, A, 5 + C]

    def model_construct(self):
        print(net_id_list)
        self.anchors = self.anchors.to(DEVICE)
        backbone, _, _ = build_model(net_id="mcunet-in4", pretrained=True)
        # Load model and loss 
        self.model = McuYolo(backbone_fn=backbone, num_classes=self.num_classes, num_anchors=len(self.anchors)).to(DEVICE)
        self.criterion = Yolov2Loss(num_classes=self.num_classes, anchors=self.anchors).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_train_loader(self):
        anchors_list = self.anchors.tolist()
        train_dataset = YoloVocDataset(
            voc_dataset=self.train_voc_raw, 
            image_size=self.image_size,
            S=self.grid_num,
            anchors=anchors_list,
            num_classes=self.num_classes,
            aug=self.aug
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.yolo_collate_fn)
        return train_loader

    def model_train(self, evaluator=None, model_path = "./saved_models", save_image=False):
        """
        Train the model, save checkpoints, and use evaluator if provided.
        """
        loss_log = {
            'total': [], 'coord': [], 'obj': [], 'class': []} 
        train_loader = self.get_train_loader()
        for epoch in range(self.epoch_num):
            self.model.train()
            # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
            # i: counts which batch you’re on (starting from 0).
            # targets: a list of batch_size elements
            # step: (data number / batchsize)
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(DEVICE)
                preds = self.model(imgs)
                loss_dict = self.criterion(preds, targets, imgs = imgs)
                loss = loss_dict['total']    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
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
                # if save_image and epoch == self.epoch_num - 1 and i % 50 == 0:  
                #     for img_idx in range(min(2, imgs.size(0))):
                #         self.save_preds_image(imgs[img_idx], preds[img_idx], epoch, i, img_idx)
            
            # Validation and checkpoint saving
            if epoch % 10 == 0:
                if evaluator:
                    evaluator.evaluate(self.model, epoch=epoch)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_log': loss_log
                }

                os.makedirs(model_path, exist_ok=True)
                suffix = "aug" if self.aug else "noaug"
                save_path = os.path.join(model_path, f"model_{self.epoch_num}_{suffix}_epoch_{epoch}.pth")
                torch.save(checkpoint, save_path)
        
        return self.model, loss_log
