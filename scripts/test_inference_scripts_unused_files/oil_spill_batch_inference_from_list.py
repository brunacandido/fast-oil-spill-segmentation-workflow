import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
model.load_state_dict(torch.load("C:/Users/Bruna/Documents/repos/oil_spill_segmentation/models/unet_oilspill_final.pth", map_location="cpu"))
model.eval()

# Diretórios
image_dir = 'C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_images'
mask_dir = 'C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_mask/Mask_oil'
output_dir = 'C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/plot_test'
os.makedirs(output_dir, exist_ok=True)

# Lista de imagens (sem extensão) a processar
# with open('C:/Users/Bruna/Documents/repos/oil_spill_segmentation/scripts/val_list.txt', 'r') as f:
#     image_names = [line.strip() for line in f if line.strip()]

image_names = ['00000', '00003', '00004', '00005']


# Transformações
transform_fn = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

# Loop sobre os arquivos
for name in image_names:
    image_path = os.path.join(image_dir, f"{name}.tif")
    mask_path = os.path.join(mask_dir, f"{name}.tif")
    output_path = os.path.join(output_dir, f"{name}_pred.png")

    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)  # canal único
        original_image = image.copy()  # para plot

    # Preprocessamento
    augmented = transform_fn(image=image)
    input_tensor = augmented['image'].unsqueeze(0)  # shape: [1, 1, 256, 256]

    # Inferência
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        pred_mask = (pred > 0.5).float().squeeze().numpy()

    # Plot
    fig, axs = plt.subplots(1, 3 if os.path.exists(mask_path) else 2, figsize=(10, 4))
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title("Imagem Original")
    axs[0].axis('off')

    if os.path.exists(mask_path):
        with rasterio.open(mask_path) as msk:
            mask = msk.read(1)
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Máscara Real")
        axs[1].axis('off')

        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title("Predição")
        axs[2].axis('off')
    else:
        axs[1].imshow(pred_mask, cmap='gray')
        axs[1].set_title("Predição")
        axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
