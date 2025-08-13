from PIL import Image
import torchvision.transforms as transforms
import torch

def process_image(image_paths, ground_truths, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    processed_images = []
    resized_gts = []

    for img_path, gt in zip(image_paths, ground_truths):
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        image_tensor = transform(image)
        processed_images.append(image_tensor)

        x1, y1, x2, y2, class_id = gt
        scale_x = image_size / orig_width
        scale_y = image_size / orig_height

        resized_gt = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, class_id]
        resized_gts.append([resized_gt])

    # all_images = torch.stack(processed_images).to(DEVICE)
    all_images = torch.stack(processed_images).to('cpu')

    return all_images, resized_gts

# Filter predictions to only keep specific detections
def filter_class_only(decoded_preds, spec_class_id):
    class_only_preds = []

    for pred_list in decoded_preds:
        boxes = []
        for box in pred_list:
            xmin, ymin, xmax, ymax, conf, class_id = box
            if int(class_id) == spec_class_id:
                boxes.append(box)
        class_only_preds.append(boxes)
    
    return class_only_preds

