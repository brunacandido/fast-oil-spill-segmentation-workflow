#!/usr/bin/env python
# coding: utf-8

# # Oil Spill Segmentation - Inference Notebook
# 
# This notebook loads a SAR image and predicts an oil spill segmentation mask using a trained U-Net model.

# In[74]:


""" # Install required libraries
!pip install segmentation-models-pytorch albumentations rasterio torch torchvision """


# In[75]:


""" import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from skimage.transform import resize """


# In[76]:


# Load SAR image (.tif)
# image_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/openEO_2017-08-10Z.tif"  # <- Update this path
# image_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_images/00054.tif"
image_path = "D:/oil_spill_images/Oil/00054.tif"
# image_path = "./SAR_raw_download.tif"
with rasterio.open(image_path) as src:
    image = src.read(1).astype(np.float32)
    profile = src.profile
    transform = src.transform


# In[77]:


# image_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/openEO_2017-08-10Z.tif"  # <- Update this path
# mask_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_mask/00054.tif"
mask_path = "D:/oil_spill_images/01_Train_Val_Oil_Spill_mask/Mask_oil/00054.tif"
#sar_original = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/satellite_results/openEO_2017-08-10Z.tif"
# mask_path = "./SAR_raw_download.tif"
with rasterio.open(mask_path) as src:
    mask_original = src.read(1).astype(np.float32)
    profile = src.profile
    transform = src.transform


# In[78]:


# Preprocessing
transform_fn = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])
augmented = transform_fn(image=image)
input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension


# In[79]:


# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
model.load_state_dict(torch.load("C:/Users/Bruna/Documents/repos/oil_spill_segmentation/models/unet_oilspill_final.pth", map_location="cpu"))
model.eval()


# In[80]:


# Predict mask
with torch.no_grad():
    pred = torch.sigmoid(model(input_tensor))
    pred_mask = (pred > 0.5).float().squeeze().numpy()


# In[81]:


# Resize prediction back to original image size
from skimage.transform import resize
original_shape = image.shape
resized_mask = resize(pred_mask, original_shape, preserve_range=True, anti_aliasing=True)
binary_mask = (resized_mask > 0.5).astype(np.uint8)


# In[82]:


# Save georeferenced segmentation mask
output_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/outputs/georeferenced_segmentation_mask"
profile.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(binary_mask, 1)
print(f"Saved segmented mask to: {output_path}")


# In[83]:


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
plt.show()



# In[84]:


""" # Convert binary mask to georeferenced polygon
import geopandas as gpd
from shapely.geometry import shape
import json
from skimage import measure

# Create contours from binary mask
contours = measure.find_contours(binary_mask, 0.5)

# Transform contours into georeferenced polygons
polygons = []
for contour in contours:
    # Transform pixel coordinates to geographic coordinates using the affine transform
    coords = [~transform * (col, row) for row, col in contour]
    polygons.append(shape({"type": "Polygon", "coordinates": [coords]}))

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=polygons, crs=profile["crs"])

# Save to file
geojson_output_path = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/outputs/geo_json"
gdf.to_file(geojson_output_path, driver="GeoJSON")
print(f"Saved segmentation polygons to: {geojson_output_path}")
 """

