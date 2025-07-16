import json
import os 
from config import DATASET_ROOT

def filter_annotation_by_visibility(coco_path, save_path, min_area=32*32, min_ratio=0.5):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    valid_image_ids = set()
    new_annotations = []

    for ann in coco_data['annotations']:
        if ann['category_id'] != 1: 
            continue
        bbox = ann['bbox']
        bbox_area = bbox[2] * bbox[3]
        seg_area = ann['area']

        # tiny object
        if bbox_area < min_area: 
            continue 

        # occlueded object
        if seg_area / bbox_area < min_ratio:
            continue

        new_annotations.append(ann)
        valid_image_ids.add(ann['image_id'])
    
    # filter images that have valid annotations 
    new_images = [img for img in coco_data['images'] if img['id'] in valid_image_ids]

    # save filtered dataset 
    filtered_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data['categories']
    }

    with open(save_path, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Saved filtered dataset with {len(new_images)} images and {len(new_annotations)} annotations.")
    print(f"Oringinal dataset contains {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")


if __name__ == "__main__":
    print("filter train dataset ...")
    # Saved filtered dataset with 42582 images and 101815 annotations.
    # Original dataset contains 118287 images and 860001 annotations.
    filter_annotation_by_visibility(
        coco_path = DATASET_ROOT + "raw/instances_train2017.json",
        save_path = DATASET_ROOT + "raw/filtered_instances_train2017.json",
    )

    print("filter validation dataset ...")
    # Saved filtered dataset with 1787 images and 4311 annotations.
    # Original dataset contains 5000 images and 36781 annotations.
    filter_annotation_by_visibility(
        coco_path = DATASET_ROOT + "raw/instances_val2017.json",
        save_path = DATASET_ROOT + "raw/filtered_instances_val2017.json",
    )