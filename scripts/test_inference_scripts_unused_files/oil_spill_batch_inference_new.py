
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from skimage.transform import resize
import glob

image_dir = "D:/oil_spill_images/Oil/"
mask_dir = "D:/oil_spill_images/01_Train_Val_Oil_Spill_mask/Mask_oil/"
images_list = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/scripts/val_list.txt"
output_dir = "D:/oil_spill_images/plots_test/dvl3_100"
model_path = "D:/oil-spill-segmentation-project/models/deepLabV3_resnet34/output/deepLabV3_resnet_oilspill_final.pth"
model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)

os.makedirs(output_dir, exist_ok=True)

with open(images_list, "r") as image_list:
    image_paths = [line.strip() for line in image_list]
    for i, image_name in enumerate(image_paths, start=1):
        total = len(image_paths)
        print(f"[{i}/{total}] Processing: {image_name}")
        image_name = image_name.strip()  
        image_path = os.path.join(image_dir, image_name + ".tif")
        mask_path = os.path.join(mask_dir, image_name + ".tif")


        with rasterio.open(image_path) as src:
            image = src.read(1).astype(np.float32)
            profile = src.profile
            transform = src.transform

        with rasterio.open(mask_path) as src:
            mask_original = src.read(1).astype(np.float32)
            profile = src.profile
            transform = src.transform

        # Preprocessing
        transform_fn = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=0, std=1),
            ToTensorV2()
        ])
        augmented = transform_fn(image=image)
        input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension

        # Load model
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # Predict mask
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred_mask = (pred > 0.5).float().squeeze().numpy()

        # Resize prediction back to original image size
        from skimage.transform import resize
        original_shape = image.shape
        resized_mask = resize(pred_mask, original_shape, preserve_range=True, anti_aliasing=True)
        binary_mask = (resized_mask > 0.5).astype(np.uint8)

        # Display original image and predicted mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original SAR Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_original, cmap="gray")
        plt.title("Original Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap="gray")
        plt.title("Predicted Oil Spill Mask")
        plt.axis("off")
        plt.tight_layout()

        plt.title(f"Inference - {image_name}")
        plot_path = os.path.join(output_dir, image_name.replace('.png', '_pred.png').replace('.jpg', '_pred.jpg'))
        plot_path = f"{output_dir}/inference_{image_name}.png"
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()