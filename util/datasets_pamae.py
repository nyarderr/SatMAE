"""
Custom PyTorch Dataset for PAMAE-ViT pre-training and fine-tuning.
 
Replaces SatMAE's fMoW-Sentinel pipeline with the PAMAE data pipeline:
  - Sentinel-2 multi-spectral patches (11 bands × 96 × 96)
  - Slope rasters (1 × 96 × 96)
  - Precipitation scalars (from metadata CSV)
  - SOC targets (for fine-tuning only)
 
Band order in TIF: B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12  (11 bands)

"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

# ──────────────────────────────────────────────────────────
# Per-band normalization statistics for Sentinel-2 (SR Harmonized)
# These are approximate pan-European statistics from LUCAS footprints.
# They will be refined during Phase 2 once a statistics pass is run
# over the full dataset.  For now they are reasonable defaults that
# keep all bands on a comparable scale after z-score normalization.
# Order: B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
# ──────────────────────────────────────────────────────────


SENTINEL2_MEAN = np.array([
    1370.0,  # B2  (Blue)
    1184.0,  # B3  (Green)
    1121.0,  # B4  (Red)
    1136.0,  # B5  (Red Edge 1)
    1264.0,  # B6  (Red Edge 2)
    1645.0,  # B7  (Red Edge 3)
    1847.0,  # B8  (NIR)
    1763.0,  # B8A (Narrow NIR)
    1973.0,  # B9  (Water Vapour)
    1732.0,  # B11 (SWIR1)
    1248.0,  # B12 (SWIR2)
], dtype=np.float32)

SENTINEL2_STD = np.array([
    633.0,   # B2
    650.0,   # B3
    712.0,   # B4
    965.0,   # B5
    949.0,   # B6
    1108.0,  # B7
    1258.0,  # B8
    1233.0,  # B8A
    1364.0,  # B9
    1310.0,  # B11
    1087.0,  # B12
], dtype=np.float32)


NUM_BANDS = len(SENTINEL2_MEAN)  # 11

# Maximum precipitation used for [0, 1] normalization.
# Approximate upper bound for annual precipitation across LUCAS footprint
# (most of Europe falls below 2500 mm/yr).
PRECIP_MAX = 2500.0
 
# Slope is divided by 90 (max possible degrees) → [0, 1]
SLOPE_DIVISOR = 90.0

class PAMAEDataset(Dataset):
    """
    PyTorch Dataset for PAMAE-ViT.
 
    Modes
    -----
    'pretrain'  : returns (image, slope, precip)
    'finetune'  : returns (image, slope, precip, soc)
 
    Parameters
    ----------
    ids_file : str
        Path to a text file with one point_id per line
        (e.g. pretrain.txt, finetune_train.txt).
    metadata_csv : str
        Path to lucas_metadata.csv with columns:
        point_id, lat, lon, soc_gkg, elevation, precip_mm
    sentinel2_dir : str
        Directory containing <point_id>.tif Sentinel-2 patches.
    slope_dir : str
        Directory containing <point_id>.tif slope rasters.
    mode : str
        'pretrain' or 'finetune'.
    band_mean : np.ndarray, optional
        Per-band mean for z-score normalization (default: SENTINEL2_MEAN).
    band_std : np.ndarray, optional
        Per-band std for z-score normalization (default: SENTINEL2_STD).

    """

    in_c = NUM_BANDS

    def __init__(self, ids_file: str, metadata_csv: str, sentinel2_dir: str, slope_dir: str, mode: str = 'pretrain', band_mean: np.ndarray = None, band_std: np.ndarray = None):
        assert mode in ['pretrain', 'finetune'], "mode must be 'pretrain' or 'finetune'"
        self.mode = mode
        self.sentinel2_dir = sentinel2_dir
        self.slope_dir = slope_dir

        # Load point IDs
        with open(ids_file, 'r') as f:
            self.point_ids = [line.strip() for line in f if line.strip()]
            assert len(self.point_ids) > 0, f"No point IDs found in {ids_file}"
        
        # Load metadata CSV
        meta = pd.read_csv(metadata_csv)
        meta['point_id'] = meta['point_id'].astype(str)  # Ensure point_id is string
        self.meta = meta.set_index('point_id')  # For fast lookup


        # Normalization stats
        band_mean = band_mean if band_mean is not None else SENTINEL2_MEAN
        band_std = band_std if band_std is not None else SENTINEL2_STD


        # Reshape for broadcasting: (C, 1, 1)
        self._mean = self.band_mean[:, None, None]
        self._std = self.band_std[:, None, None]


    
    def __len__(self):
        return len(self.point_ids)
    
    def __getitem__(self, idx:int):
        point_id = self.point_ids[idx]
        
        # Load Sentinel-2 patch
        S2_path = os.path.join(self.sentinel2_dir, f"{point_id}.tif")
        image = self.load_tif(S2_path)  # (C, H, W) float32
        # Z-score normalize
        image = (image - self._mean)/(self._std + 1e-8)
        image = torch.from_numpy(image)  # Convert to tensor


        # Load slope raster
        slope_path = os.path.join(self.slope_dir, f"{point_id}.tif")
        slope = self.load_tif(slope_path)  # (1, H, W) float32
        slope = slope / SLOPE_DIVISOR  # Normalize to [0,1]
        slope = np.clip(slope, 0.0, 1.0)  # Ensure within [0,1]
        slope = torch.from_numpy(slope)


        # Load metadata
        meta_row = self.meta.loc[point_id]
        precip = meta_row['precip_mm'] / PRECIP_MAX  # Normalize to [0,1]
        precip = np.clip(precip, 0.0, 1.0)  # Ensure within [0,1]
        precip = torch.tensor(precip, dtype=torch.float32)


        if self.mode == 'pretrain':
            return image, slope, precip
        
        # For fine-tuning, also return SOC target (log-transformed)
        soc_raw = np.float32(meta_row['soc_gkg'])
        soc_log = np.log1p(soc_raw) 
        soc = torch.tensor(soc_log, dtype=torch.float32)

        return image, slope, precip, soc


    ## ──────────────────────────────────────────────────────────
    ## Helper functions
    ## ──────────────────────────────────────────────────────────

    @staticmethod
    def load_tif(path:str) -> np.ndarray:
        """Read a GeoTIFF and return as float32 array of shape (C, H, W)."""
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
            arr = arr.astype(np.float32)  # Convert to float32
        return arr
    
    def __repr__(self):
        return f"PAMAEDataset(mode={self.mode}, num_samples={len(self)}, bands = {self.in_c}, sentinel2_dir='{self.sentinel2_dir}', slope_dir='{self.slope_dir}')"
    



## Builder function for convenience
def build_pamae_dataset(args, mode:str = "pretrain"):
    """
    Construct a PAMAEDataset from argparse args.
 
    Expected args attributes:
        args.data_dir       : root of the dataset/ directory
        args.ids_file       : path to the IDs text file
                              (overrides default pretrain.txt / finetune_*.txt)
    """

    data_dir = args.data_dir

    metadata_csv = os.path.join(data_dir, "lucas_metadata.csv")
    sentinel2_dir = os.path.join(data_dir, "sentinel2")
    slope_dir = os.path.join(data_dir, "slope")


    # Determine IDs file based on mode
    if hasattr(args, "ids_file") and args.ids_file is not None:
        ids_file = args.ids_file
    else:
        # Default: pretrain.txt for pretrain mode
        if mode == "pretrain":
            ids_file = os.path.join(data_dir, "pretrain.txt")
        else:
            ids_file = os.path.join(data_dir, "finetune_train.txt")

    dataset = PAMAEDataset(
        ids_file=ids_file,
        metadata_csv=metadata_csv,
        sentinel2_dir=sentinel2_dir,
        slope_dir=slope_dir,
        mode=mode
    )

    return dataset



        
        

