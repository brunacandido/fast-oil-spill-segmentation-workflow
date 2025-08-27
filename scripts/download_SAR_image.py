"""
Download Sentinel-1 SAR data from openEO and save as GeoTIFF.
"""

import os
import rasterio
import matplotlib.pyplot as plt


def run_download(coordinates, temporal_extent, outputfile, connection):
    """
    Download Sentinel-1 SAR images from openEO based on given parameters.

    Parameters
    ----------
    coordinates : list
        List of coordinates [lon, lat] forming the polygon of the Area of Interest (AOI).
    temporal_extent : list
        Time interval in the format ["YYYY-MM-DD", "YYYY-MM-DD"].
    outputfile : str
        Path to the output GeoTIFF file.
    """

    # Define the bounding box from polygon
    lons = [p[0] for p in coordinates]
    lats = [p[1] for p in coordinates]
    bbox = {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats)
    }
    print('bbox',bbox)
    # Load Sentinel-1 GRD collection
    s1_image_original = connection.load_collection(
        "SENTINEL1_GRD",
        spatial_extent=bbox,
        temporal_extent=temporal_extent,
        bands=["VV"]
    )

    print(s1_image_original,'s1_image_original')

    # Apply backscatter calibration
    s1_image = s1_image_original.sar_backscatter(coefficient="sigma0-ellipsoid")

    # %%
    # Export the processed image as GeoTIFF
    job = s1_image.execute_batch(
        title="Sentinel1_Processed_TIFF",
        outputfile=outputfile,
        out_format="GTIFF"
    )

    # %%
    directory = os.path.dirname(outputfile)
    results = job.get_results()
    results.download_files(directory) 

    with rasterio.open(outputfile) as src:
        image = src.read(1)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title("Sentinel-1 SAR Image")
    plt.axis('off')
    plt.show()
    # Return path for chaining in pipeline
    return outputfile


if __name__ == "__main__":
    # Example call (replace with real values)
    coords = [
        [34.2430882356336, 34.1860792481070],
        [34.3797790095640, 34.1660375298848],
        [34.3546654835910, 34.0509074628334],
        [34.2179747096606, 34.0709491810556],
        [34.2430882356336, 34.1860792481070]
    ]
    temporal = ["2019-06-18", "2019-06-19"]
    output = "../data/satellite_results/1_Sentinel1_from_openEO.tif"

    run_download(coords, temporal, output)

