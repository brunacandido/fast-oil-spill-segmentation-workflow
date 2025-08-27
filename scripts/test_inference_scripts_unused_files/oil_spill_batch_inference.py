
import os
import glob
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image_path = "D:/oil_spill_images/Oil/00054.tif"
mask_path = "D:/oil_spill_images/01_Train_Val_Oil_Spill_mask/Mask_oil/00054.tif"
image_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))

transform_fn = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])
augmented = transform_fn(image=image)
input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension

input_dir = 'path_to_input_images'  
output_dir = 'path_to_output_plots'
os.makedirs(output_dir, exist_ok=True)

# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
model.load_state_dict(torch.load("C:/Users/Bruna/Documents/repos/oil_spill_segmentation/models/unet_oilspill_final.pth", map_location="cpu"))
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


# Loop sobre as imagens
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)[0]

    # Plot sem mostrar
    plt.figure()
    plt.imshow(output.squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Inference - {image_name}")
    plot_path = os.path.join(output_dir, image_name.replace('.png', '_pred.png').replace('.jpg', '_pred.jpg'))
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close()
