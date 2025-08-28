"""
Preprocess Sentinel-1 SAR data.
"""

import os
import rasterio
import numpy as np

def run_treatment(input_file, output_folder):
    """
    Preprocess Sentinel-1 SAR image (normalization and cleanup).

    Parameters
    ----------
    input_file : str
        Path to the downloaded GeoTIFF file.
    output_folder : str
        Directory where processed results will be saved.

    Returns
    -------
    str
        Path to the processed file.
    """
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "processed_sentinel1.tif")

    with rasterio.open(input_file) as src:
        profile = src.profile
        data = src.read()  # all bands

        # Example preprocessing: normalize values between 0-1
        data = data.astype("float32")
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        norm_data = (data - data_min) / (data_max - data_min + 1e-6)

        # Update profile
        profile.update(dtype=rasterio.float32, count=norm_data.shape[0])

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(norm_data.astype(rasterio.float32))

    return output_file


if __name__ == "__main__":
    # Example call
    run_treatment("../data/satellite_results/1_Sentinel1_from_openEO.tif", "results/")
