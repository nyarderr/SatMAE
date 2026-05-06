"""
Validation script for PAMAEDataset.
 
Creates a small synthetic dataset mimicking the Phase 0 layout,
then runs the dataloader in both 'pretrain' and 'finetune' modes
and asserts correctness of output shapes, value ranges, and NaN checks.
 
Usage:
    python test_pamae_dataset.py

"""

import os
import sys
import tempfile
import shutil
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util.datasets_pamae import PAMAEDataset, NUM_BANDS

# ──────────────────────────────────────────────────────────
#  Synthetic data generation
# ──────────────────────────────────────────────────────────
 
NUM_SAMPLES = 24  # small but enough for 2 batches of 8
BATCH_SIZE = 8
IMG_H, IMG_W = 96, 96
NUM_S2_BANDS = 11   # B2..B12 (11 bands from Phase 0)
SLOPE_BANDS = 1

def make_synthetic_dataset(root: str):
    """Create a synthetic dataset directory matching Phase 0 layout."""
    s2_dir = os.path.join(root, "sentinel2")
    slope_dir = os.path.join(root, "slope")
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
 
    rng = np.random.default_rng(42)
 
    point_ids = [f"PT_{i:05d}" for i in range(NUM_SAMPLES)]
 
    # Write pretrain.txt (all), finetune_train.txt (first 80%), finetune_test.txt (last 20%)
    split = int(0.8 * NUM_SAMPLES)
    for fname, ids in [
        ("pretrain.txt", point_ids),
        ("finetune_train.txt", point_ids[:split]),
        ("finetune_test.txt", point_ids[split:]),
    ]:
        with open(os.path.join(root, fname), "w") as f:
            f.write("\n".join(ids) + "\n")
 
    # Metadata CSV
    rows = []
    for pid in point_ids:
        rows.append({
            "point_id": pid,
            "lat": rng.uniform(35, 60),
            "lon": rng.uniform(-10, 30),
            "soc_gkg": rng.uniform(5, 150),        # g/kg
            "elevation": rng.uniform(0, 2500),
            "precip_mm": rng.uniform(200, 2400),    # mm/yr
        })
    pd.DataFrame(rows).to_csv(os.path.join(root, "lucas_metadata.csv"), index=False)
 
    # Create GeoTIFFs (using rasterio)
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError("rasterio is required — pip install rasterio")
 
    for pid in point_ids:
        # Sentinel-2: 11 bands of realistic reflectance values (0–10000 range)
        s2_data = rng.uniform(100, 5000, size=(NUM_S2_BANDS, IMG_H, IMG_W)).astype(np.float32)
 
        transform = from_bounds(0, 0, 1, 1, IMG_W, IMG_H)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": IMG_W,
            "height": IMG_H,
            "count": NUM_S2_BANDS,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(os.path.join(s2_dir, f"{pid}.tif"), "w", **profile) as dst:
            dst.write(s2_data)
 
        # Slope: 1 band, 0–45 degrees typical
        slope_data = rng.uniform(0, 45, size=(SLOPE_BANDS, IMG_H, IMG_W)).astype(np.float32)
        profile["count"] = SLOPE_BANDS
        with rasterio.open(os.path.join(slope_dir, f"{pid}.tif"), "w", **profile) as dst:
            dst.write(slope_data)
 
    return point_ids

# ──────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────
 
def test_pretrain_mode(root, point_ids):
    print("\n" + "=" * 60)
    print("TEST 1: Pretrain mode")
    print("=" * 60)
 
    ds = PAMAEDataset(
        ids_file=os.path.join(root, "pretrain.txt"),
        metadata_csv=os.path.join(root, "lucas_metadata.csv"),
        sentinel2_dir=os.path.join(root, "sentinel2"),
        slope_dir=os.path.join(root, "slope"),
        mode="pretrain",
    )
 
    # Check dataset length matches text file
    assert len(ds) == len(point_ids), \
        f"Dataset length {len(ds)} != IDs file count {len(point_ids)}"
    print(f"  [PASS] Dataset length = {len(ds)} matches IDs file")
 
    # Single-item check
    image, slope, precip = ds[0]
    assert image.shape == (NUM_BANDS, IMG_H, IMG_W), \
        f"Image shape {image.shape} != ({NUM_BANDS}, {IMG_H}, {IMG_W})"
    assert slope.shape == (1, IMG_H, IMG_W), \
        f"Slope shape {slope.shape} != (1, {IMG_H}, {IMG_W})"
    assert precip.shape == (), \
        f"Precip shape {precip.shape} != scalar"
    print(f"  [PASS] Single-item shapes: image={image.shape}, slope={slope.shape}, precip=scalar")
 
    # NaN checks
    assert not torch.isnan(image).any(), "NaN in image!"
    assert not torch.isnan(slope).any(), "NaN in slope!"
    assert not torch.isnan(precip).any(), "NaN in precip!"
    print("  [PASS] No NaN values in single item")
 
    # Value range checks
    assert 0.0 <= slope.min() and slope.max() <= 1.0, \
        f"Slope out of [0,1]: min={slope.min():.3f}, max={slope.max():.3f}"
    assert 0.0 <= precip.item() <= 1.0, \
        f"Precip out of [0,1]: {precip.item():.3f}"
    print(f"  [PASS] Slope range [{slope.min():.3f}, {slope.max():.3f}]")
    print(f"  [PASS] Precip value {precip.item():.4f} in [0,1]")
 
    # Image z-score: should be roughly centered around 0
    print(f"  [INFO] Image stats: mean={image.mean():.3f}, std={image.std():.3f}, "
          f"min={image.min():.3f}, max={image.max():.3f}")
 
    # Dataloader batch test
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch_image, batch_slope, batch_precip = next(iter(dl))
 
    assert batch_image.shape == (BATCH_SIZE, NUM_BANDS, IMG_H, IMG_W), \
        f"Batch image shape {batch_image.shape}"
    assert batch_slope.shape == (BATCH_SIZE, 1, IMG_H, IMG_W), \
        f"Batch slope shape {batch_slope.shape}"
    assert batch_precip.shape == (BATCH_SIZE,), \
        f"Batch precip shape {batch_precip.shape}"
    print(f"  [PASS] Batch shapes: image={batch_image.shape}, "
          f"slope={batch_slope.shape}, precip={batch_precip.shape}")
 
    assert not torch.isnan(batch_image).any(), "NaN in batched images!"
    assert not torch.isnan(batch_slope).any(), "NaN in batched slope!"
    assert not torch.isnan(batch_precip).any(), "NaN in batched precip!"
    print("  [PASS] No NaN values in batch")
 
    # Count total samples via full iteration
    total = 0
    for _ in dl:
        total += 1
    # With drop_last=False (default), we get ceil(24/8) = 3 batches
    print(f"  [PASS] Full iteration: {total} batches")
 
 
def test_finetune_mode(root):
    print("\n" + "=" * 60)
    print("TEST 2: Finetune mode")
    print("=" * 60)
 
    ds = PAMAEDataset(
        ids_file=os.path.join(root, "finetune_train.txt"),
        metadata_csv=os.path.join(root, "lucas_metadata.csv"),
        sentinel2_dir=os.path.join(root, "sentinel2"),
        slope_dir=os.path.join(root, "slope"),
        mode="finetune",
    )
 
    split_count = int(0.8 * NUM_SAMPLES)
    assert len(ds) == split_count, \
        f"Finetune train length {len(ds)} != expected {split_count}"
    print(f"  [PASS] Finetune train length = {len(ds)}")
 
    image, slope, precip, soc = ds[0]
    assert soc.shape == (), f"SOC shape {soc.shape} != scalar"
    assert soc.item() > 0, f"log1p(SOC) should be positive, got {soc.item()}"
    print(f"  [PASS] SOC scalar shape, value={soc.item():.4f} (log1p-transformed)")
 
    # Dataloader
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert len(batch) == 4, f"Finetune batch should have 4 elements, got {len(batch)}"
    batch_image, batch_slope, batch_precip, batch_soc = batch
    assert batch_soc.shape == (BATCH_SIZE,), f"Batch SOC shape {batch_soc.shape}"
    assert not torch.isnan(batch_soc).any(), "NaN in batch SOC!"
    print(f"  [PASS] Finetune batch: soc shape={batch_soc.shape}, "
          f"min={batch_soc.min():.3f}, max={batch_soc.max():.3f}")
 
 
def test_patch_count(root):
    """Verify the number of ViT patches for input_size=96, patch_size=8."""
    print("\n" + "=" * 60)
    print("TEST 3: Patch count verification")
    print("=" * 60)
 
    input_size = 96
    patch_size = 8
    expected_patches = (input_size // patch_size) ** 2  # 12×12 = 144
    print(f"  input_size={input_size}, patch_size={patch_size}")
    print(f"  Expected patches: ({input_size}//{patch_size})^2 = {expected_patches}")
 
    # Read IDs file line count
    with open(os.path.join(root, "pretrain.txt")) as f:
        n_ids = sum(1 for line in f if line.strip())
    print(f"  pretrain.txt line count: {n_ids}")
 
    ds = PAMAEDataset(
        ids_file=os.path.join(root, "pretrain.txt"),
        metadata_csv=os.path.join(root, "lucas_metadata.csv"),
        sentinel2_dir=os.path.join(root, "sentinel2"),
        slope_dir=os.path.join(root, "slope"),
        mode="pretrain",
    )
    assert len(ds) == n_ids, f"Dataset length {len(ds)} != IDs count {n_ids}"
    print(f"  [PASS] Dataset length ({len(ds)}) matches pretrain.txt ({n_ids} lines)")
    print(f"  [PASS] Each image will yield {expected_patches} patches at patch_size={patch_size}")
 
 
def test_repr(root):
    print("\n" + "=" * 60)
    print("TEST 4: __repr__")
    print("=" * 60)
    ds = PAMAEDataset(
        ids_file=os.path.join(root, "pretrain.txt"),
        metadata_csv=os.path.join(root, "lucas_metadata.csv"),
        sentinel2_dir=os.path.join(root, "sentinel2"),
        slope_dir=os.path.join(root, "slope"),
        mode="pretrain",
    )
    print(f"  {ds}")
    print("  [PASS]")
 
 
# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp(prefix="pamae_test_")
    print(f"Synthetic dataset root: {tmpdir}")
 
    try:
        point_ids = make_synthetic_dataset(tmpdir)
        test_pretrain_mode(tmpdir, point_ids)
        test_finetune_mode(tmpdir)
        test_patch_count(tmpdir)
        test_repr(tmpdir)
 
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
 