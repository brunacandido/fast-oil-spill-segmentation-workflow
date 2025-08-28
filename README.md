# fast-oil-spill-segmentation-workflow  

A reproducible workflow for detecting **oil spills in Sentinel-1 SAR imagery** using deep learning.  
This repository provides:  

- **Notebooks** → Step-by-step processing and full pipelines  
- **Scripts** → Modular Python functions for automation  
- **Pre-trained Models** → Ready-to-use weights for inference  

---

## 🔎 Overview  

Oil spills are a major environmental hazard, and monitoring large ocean areas requires scalable methods.  
This workflow leverages **Sentinel-1 Synthetic Aperture Radar (SAR)** data and **deep learning segmentation models** (U-Net, FPN, DeepLabV) to detect oil spills automatically.  

It supports both **educational exploration** (via notebooks) and **operational application** (via scripts and pipelines).  The output is a georreferenced oil spill polygon, which can be used in oil spill modeling.

---

## 🧾 Notebooks  

1. **Download Sentinel-1 SAR**  
   - Uses [openEO](https://openeo.org/) API to query and download Sentinel-1 data (VV polarization).  

2. **Preprocessing**  
   - Converts Sentinel-1 SAR imagery from **linear power to decibel (dB) scale**  
   - Handles invalid pixel values by replacing non-positive values with the smallest positive value in the image  
   - Saves the converted image as a new GeoTIFF and provides a visualization (before/after conversion)  

3. **Oil Spill Inference**  
   - Runs trained models on preprocessed SAR images  
   - Produces binary segmentation masks (oil / no oil)  

4. **Pipelines**  
   - End-to-end workflows combining all steps  
   - Four examples with different Sentinel-1 images  

5. **Visualization**  
   - `visualize_tif.ipynb` for visualizing SAR `.tif` images easily

---

## 🧠 Trained Models  

Pre-trained weights are included for fast inference:  

- **FPN (EfficientNet-b0)**  
- **DeepLabV3 (ResNet34)**  
- **PAN (ResNet34)**  

---

## 🚀 Getting Started  

### Setup environment  
```bash
conda env create -f environment.yml
conda activate oilspill-env