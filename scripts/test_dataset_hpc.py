import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# --- Step 1: List image and mask files ---
image_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/test_data/Oil"
mask_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/test_data/Masks"

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

# Ensure filenames match
matched_image_paths = []
matched_mask_paths = []
unmatched_images = []

for img_name in image_files:
    if img_name in mask_files:
        matched_image_paths.append(os.path.join(image_dir, img_name))
        matched_mask_paths.append(os.path.join(mask_dir, img_name))
    else:
        unmatched_images.append(img_name)

print(f"Matched {len(matched_image_paths)} image-mask pairs.")
print(f"{len(unmatched_images)} images have no matching mask.")

# --- Step 2: Define Dataset ---
class SARDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            with rasterio.open(img_path) as img_src:
                image = img_src.read(1).astype(np.float32)

            with rasterio.open(mask_path) as mask_src:
                mask = mask_src.read(1).astype(np.float32)
                mask = (mask > 0).astype(np.float32)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]

            return image, mask, os.path.basename(img_path)

        except Exception as e:
            raise RuntimeError(f"Error processing {img_path}: {e}")

# --- Step 3: Set transform ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

# --- Step 4: Check for problematic files ---
dataset = SARDataset(matched_image_paths, matched_mask_paths, transform=transform)
problematic_images = []

for i in tqdm(range(len(dataset)), desc="Checking image-mask pairs"):
    try:
        _, _, filename = dataset[i]
    except Exception as e:
        print(e)
        problematic_images.append(os.path.basename(dataset.image_paths[i]))

# Add unmatched image names to the list of problems
problematic_images.extend(unmatched_images)

# --- Step 5: Save problematic image names to a file ---
with open("problematic_images.txt", "w") as f:
    for name in problematic_images:
        f.write(name + "\n")

print(f"Done. {len(problematic_images)} problematic images saved to 'problematic_images.txt'.")