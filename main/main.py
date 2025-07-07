import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
import torch

import preprocess
import model
import ast

# Read imagenet_labels
with open("imagenet_labels.txt", "r") as f:
    class_dict = ast.literal_eval(f.read())


train_dataset = preprocess.get_dataset("validation")
model = model.get_model()

for idx, sample in enumerate(train_dataset):
    print("Image path: ", sample.filepath)

    img = Image.open(sample.filepath)
    w, h = img.size

    # Convert to RGB
    # Some Images Are Not RGB by Default
    img = img.convert("RGB") 

    # Resize and normalise
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor() # HWC -> CHW 
    ])

    tensor_img = transform(img)
    # Convert tensor back to PIL image
    resize_img = transforms.functional.to_pil_image(tensor_img)
    # Save to disk (compressed)
    resize_img.save(f"./images/resize_image_{idx}.png")  

    # Input tensor for training [3,H,W] -> [1,3,H,W]
    input_tensor = transform(img).unsqueeze(0) 

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

        # print(output)
        probs = torch.softmax(output, dim=1) # [1, num_classes]
        pred_class = torch.argmax(probs).item()
        print(pred_class)
        print(class_dict[pred_class])

    # # Start a plot
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)

    # # Draw detection boxes
    # if sample.ground_truth:
    #     detections = sample.ground_truth.detections
    # else:
    #     detections = []
    
    # for det in detections:
    #     label = det.label
    #     bbox = det.bounding_box # normalized [x, y, width, height]

    #     x = bbox[0] * w
    #     y = bbox[1] * h 
    #     width = bbox[2] * w 
    #     height = bbox[3] * h 

    #     # add bounding box on the image
    #     rect = patches.Rectangle(
    #         (x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
    #     ax.add_patch(rect)
    #     ax.text(x, y, label, color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    # ax.axis("off")

    # # Save figure 
    # save_path = f"./images/image_{idx}.png"
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)  # Free memory
