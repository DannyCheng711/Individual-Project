import os
import json
import gzip
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

CATEGORY = "motorcycle"
BASE_DIR = f"./{CATEGORY}"
OUTPUT_DIR = f"./{CATEGORY}/_selection"
MAX_SEQUENCES = 20
FRAMES_PER_SEQ = 10
MASK_COVERAGE_RANGE = (0.2, 0.7)
DISTANCE_RANGE = (3.0, 7.0)

class CO3DProcessor:
    def __init__(self, category, base_dir, output_dir):
        self.category = category
        self.base_dir = base_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Helpers 
    @staticmethod
    def load_jgz(path):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def compute_mask_coverage(mask_path): 
        mask = Image.open(mask_path).convert("L") # convert the image to 8-bit grayscale
        mask_np = np.array(mask) > 0 # pixel contains object (not pure black)
        return mask_np.sum() / mask_np.size

    @staticmethod
    # T = [x,y,z], its Euclidean distance to the object center sqrt(x ** 2 + y ** 2 + z ** 2)
    def compute_distance(T):
        return np.linalg.norm(T)

    @staticmethod
    def select_diverse_frames(frames, num_frames):
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    @staticmethod
    def safe_copy(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def extract_bbox_from_mask(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask) > 0
        if mask_np.sum() == 0:
            return None
        
        # returns the coordinates of all pixels where mask_np == True.
        # ys -> all the row indices
        # xs -> all the column indices
        ys, xs = np.where(mask_np)

        return [
            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        ]  # x_min, y_min, x_max, y_max
        

    # Step 1: Generate raw dataset
    def generate_raw_data(self, max_sequences=10, frames_per_seq=10, coverage_range=(0.2,0.7), distance_range=(3,7)):
        frame_annotations = self.load_jgz(os.path.join(self.base_dir, "frame_annotations.jgz"))
        sequence_annotations = self.load_jgz(os.path.join(self.base_dir, "sequence_annotations.jgz"))
        seq_meta = {seq["sequence_name"]: seq for seq in sequence_annotations}

        selected_sequences = {}

        for frame in tqdm(frame_annotations, desc="Processing frames"):
            seq_name = frame["sequence_name"]
            category = seq_meta[seq_name]["category"]
            # skip the category which not in the selected category
            if category != self.category:
                continue
    
            mask_path = os.path.join("./", frame['mask']['path'])
            image_path = os.path.join("./", frame['image']['path'])

            # compute coverage and distance
            coverage = self.compute_mask_coverage(mask_path)
            distance = self.compute_distance(frame['viewpoint']['T'])
    
            if coverage > MASK_COVERAGE_RANGE[1] or coverage < MASK_COVERAGE_RANGE[0]:
                continue
            if distance > DISTANCE_RANGE[1] or distance < DISTANCE_RANGE[0]:
                continue
    

            if seq_name not in selected_sequences:
                selected_sequences[seq_name] = []
            selected_sequences[seq_name].append({
                'frame_id': frame['frame_number'],
                'image_path': image_path,
                'mask_path': mask_path,
                'coverage': coverage,
                'distance': distance
            })
        
        # Sort & select sequences
        final_selection = []
        selected_sequences_tuple = sorted(selected_sequences.items(), key=lambda x: -len(x[1]))[:max_sequences]

        # frames : frame list
        for seq_name, frames in selected_sequences_tuple:
            frames = sorted(frames, key = lambda x: x['frame_id'])
            chosen = self.select_diverse_frames(frames, frames_per_seq)
            # add seq_name tag and category tag
            for frame in chosen:
                frame['seq_name']= seq_name
                frame['cat'] = CATEGORY
                final_selection.append(frame)

        # copy files
        manifest = []
        for item in tqdm(final_selection, desc="Copying frames"):
            out_img = os.path.join(OUTPUT_DIR, CATEGORY, item['seq_name'], f"{item['seq_name']}_{item['frame_id']}.jpg")
            out_mask = os.path.join(OUTPUT_DIR, CATEGORY, item['seq_name'], f"{item['seq_name']}_{item['frame_id']}_mask.png")
            self.safe_copy(item['image_path'], out_img)
            self.safe_copy(item['mask_path'], out_mask)

            manifest.append({
                'image': out_img,
                'mask': out_mask,
                'coverage': item['coverage'],
                'distance': item['distance'],
                'seq_name': item['seq_name'],
                'category': item['cat']
            })

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved {len(final_selection)} frames to {self.output_dir}")
        return manifest_path

    # Step 2: Create BBoxes
    def create_bboxes(self, manifest_path, add_class=True):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for item in tqdm(manifest, desc="Generating BBoxes"):
            bbox = self.extract_bbox_from_mask(item['mask'])
            item['bbox'] = bbox if bbox else [0, 0, 0, 0]
            if add_class: 
                item['class'] = self.category
        
        bbox_path = os.path.join(os.path.dirname(manifest_path), "manifest_with_bboxes.json")
        with open(bbox_path, 'w') as f:
            json.dump(manifest, f, indent = 2)

        print(f"Bounding boxes with class saved to {bbox_path}")
        return bbox_path
    
    # Step 3: Create Images with Occlusion
    def random_rect(self, bbox, rate):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        area = w * h
        target_area = area * rate
        cut_w = int(np.sqrt(target_area))
        cut_h = int(target_area / max(cut_w, 1))
        cut_w = min(cut_w, w) # cutout width cannot extend image width
        cut_h = min(cut_h, h)
        # select the cutout coordinate
        cut_x = np.random.randint(xmin, xmax - cut_w + 1)
        cut_y = np.random.randint(ymin, ymax - cut_h + 1)
        return (cut_x, cut_y, cut_x + cut_w, cut_y + cut_h)

    def apply_cutout(self, img, rect):
        x1, y1, x2, y2 = rect
        occluded = img.copy()
        occluded[y1:y2, x1:x2] = 0 # black out
        return occluded

    def create_occlusion_images(self, manifest_path, occlusion_rates):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for item in tqdm(manifest, desc="Generating Occlusion Images"):
            image = cv2.imread(item['image'])
            bbox = item['bbox']

            for rate in occlusion_rates:

                rect = self.random_rect(bbox, rate)
                occ = self.apply_cutout(image, rect)
                occ_path = os.path.join(
                    self.output_dir, item['seq_name'] ,f"{os.path.splitext(os.path.basename(item['image']))[0]}_occ{int(rate*100)}.jpg")
                cv2.imwrite(occ_path, occ)
                item[f"occ{int(rate*100)}"] = occ_path

        occ_manifest_path = os.path.join(os.path.dirname(manifest_path), f"manifest_with_occ.json")
        with open(occ_manifest_path, 'w') as f:
            json.dump(manifest, f, indent = 2)

            



if __name__ == "__main__":
    processor = CO3DProcessor(
        category="car",
        base_dir = "./co3d/car", 
        output_dir= "./co3d/car"
    )

    # manifest_path = os.path.join(
    #    os.path.dirname("./co3d/car/"), "manifest.json")

    # processor.create_bboxes(manifest_path=manifest_path)

    manifest_path_bbox = os.path.join(
        os.path.dirname("./co3d/car/"), "manifest_with_bboxes.json")
    processor.create_occlusion_images(manifest_path=manifest_path_bbox, occlusion_rates=[0.3, 0.5, 0.7])