
import os
import csv
import random
import torch
import rasterio
import numpy as np
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    image_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/test_data/Oil"
    mask_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/test_data/Masks"
    output_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/test_data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    matched_image_paths, matched_mask_paths = [], []

    for img_name in image_files:
        if img_name in mask_files:
            matched_image_paths.append(os.path.join(image_dir, img_name))
            matched_mask_paths.append(os.path.join(mask_dir, img_name))

    print(f"Matched {len(matched_image_paths)} image-mask pairs.")

    class SARDataset(Dataset):
        def __init__(self, image_paths, mask_paths, transform=None):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            with rasterio.open(self.image_paths[idx]) as img_src:
                image = img_src.read(1).astype(np.float32)
            with rasterio.open(self.mask_paths[idx]) as mask_src:
                mask = mask_src.read(1).astype(np.float32)
                mask = (mask > 0).astype(np.float32)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]
            return image, mask

    transform = A.Compose([
        A.Resize(256, 256),   # we can also try 384×384 ou 512×512 
        A.Normalize(mean=0, std=1),
        ToTensorV2()
    ])

    full_dataset = SARDataset(matched_image_paths, matched_mask_paths, transform=transform)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.to(device)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "ValLoss", "IoU", "Precision", "Recall", "F1"])

    num_epochs = 10

    train_losses = []
    val_losses = []
    ious = []
    precisions = []
    recalls = []
    f1s = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train"):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        preds_flat = np.concatenate(all_preds).reshape(-1)
        targets_flat = np.concatenate(all_targets).reshape(-1)

        iou = jaccard_score(targets_flat, preds_flat)
        precision = precision_score(targets_flat, preds_flat)
        recall = recall_score(targets_flat, preds_flat)
        f1 = f1_score(targets_flat, preds_flat)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("=" * 60)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, iou, precision, recall, f1])

        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    final_model_path = os.path.join(output_dir, "unet_oilspill_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved in:", final_model_path)

        # Plotar e salvar as curvas de métricas
    epochs = list(range(1, len(train_losses) + 1))

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Métricas
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ious, label="IoU")
    plt.plot(epochs, f1s, label="F1 Score")
    plt.plot(epochs, precisions, label="Precision")
    plt.plot(epochs, recalls, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "metrics_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()