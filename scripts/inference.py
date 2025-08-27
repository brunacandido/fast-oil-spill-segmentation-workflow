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
from shapely.geometry import Polygon
import geopandas as gpd
from skimage import measure
import sys
sys.path.append(os.path.abspath(".."))
from scripts.download_SAR_image import run_download
from scripts.get_unique_path import get_unique_path


def run_inference(processed_file, model_name, output_folder):
    """
    Run oil spill segmentation inference on a SAR image using a trained DeepLabV3+ model.

    Args:
        processed_file (str): Path to the preprocessed SAR .tif file.
        model_path (str): Path to the trained model (.pth).
        output_folder (str): Directory where outputs will be saved.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Load preprocessed SAR image
    with rasterio.open(processed_file) as src:
        image = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs

    # Preprocess image for model
    transform_fn = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=0, std=1),
        ToTensorV2()
    ])
    augmented = transform_fn(image=image)
    input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension

    # Load model
    MODEL_DIR = "../trained_models"

    # models dictionary
    MODEL_CONFIGS = {
        "deeplabv3": {
            "path": os.path.join(MODEL_DIR, "deepLabV3_resnet_oilspill_final.pth"),
            "class": smp.DeepLabV3Plus,
            "params": {
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 1,
                "classes": 1,
            },
        },
        "fpn": {
            "path": os.path.join(MODEL_DIR, "FPN_efficientnet-b0_oilspill_final.pth"),
            "class": smp.FPN,
            "params": {
                "encoder_name": "efficientnet-b0",
                "encoder_weights": "imagenet",
                "in_channels": 1,
                "classes": 1,
            },
        },
        "pan": {
            "path": os.path.join(MODEL_DIR, "pan_resnet_oilspill_final.pth"),
            "class": smp.PAN,
            "params": {
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 1,
                "classes": 1,
            },
        },
    }

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found. Options: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model_class = config["class"]
    params = config["params"]
    model_path = config["path"]

    # Instancia o modelo
    model = model_class(**params)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict mask
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        pred_mask = (pred > 0.5).float().squeeze().numpy()

    # Resize prediction back to original image size
    original_shape = image.shape
    resized_mask = resize(pred_mask, original_shape, preserve_range=True, anti_aliasing=True)
    binary_mask = (resized_mask > 0.5).astype(np.uint8)

    # Save georeferenced segmentation mask
    mask_output_path = os.path.join(output_folder, "georeferenced_segmentation_mask.tif")
    unique_output_path_mask = get_unique_path(mask_output_path)

    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(unique_output_path_mask, 'w', **profile) as dst:
        dst.write(binary_mask, 1)
    print(f"✅ Saved segmented mask to: {unique_output_path_mask}")

    # Export georeferenced polygons
    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []
    for contour in contours:
        coords = [rasterio.transform.xy(transform, row, col) for row, col in contour]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if poly.is_valid:
            polygons.append(poly)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    geojson_output_path = os.path.join(output_folder, "georreferenced_segmentation.geojson")
    unique_output_path = get_unique_path(geojson_output_path)
    gdf.to_file(unique_output_path, driver="GeoJSON")
    print(f"✅ Saved polygons to: {geojson_output_path}")

    # Plot image + prediction
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original SAR Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Predicted Oil Spill Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return mask_output_path, geojson_output_path
