from libs.autoencoder import get_model
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
from pathlib import Path
import glob
from majortom.NMajorTOM import NMajorTOM
import pyarrow.parquet as pq
import geopandas as gpd
import pandas as pd
from majortom.coverage_vis import get_coveragemap

torch.manual_seed(0)
np.random.seed(0)

PATCH_SIZE = 256
GRID_SIZE = 4  # 4x4 grid of patches

SATELLITE_CONFIGS = {
    'S2L2A': {
        'tif_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'cloud_mask'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),  # Crop to 1024x1024
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'S2L1C': {
        'tif_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'cloud_mask'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'S1RTC': {
        'tif_bands': ['vv', 'vh'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'DEM': {
        'tif_bands': ['DEM', 'compressed'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.Resize(1068), # First, interpolate to match the resolution of the other modalities (1068x1068)
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    }
}

def fix_crs(df):
    if df['crs'].iloc[0].startswith('EPSG:EPSG:'):
        df['crs'] = df['crs'].str.replace('EPSG:EPSG:', 'EPSG:', regex=False)
    return df

def load_metadata(path):
    df = pq.read_table(path).to_pandas()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df.timestamp)
    df = fix_crs(df)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.centre_lon, df.centre_lat), crs=df.crs.iloc[0]
    )
    return gdf

def process_satellite(subset_path, satellite_types, bands_per_type, ratio_train_test, seed):
    """Process multiple satellite types simultaneously while ensuring they're paired"""
    modalities = {}
    filtered_dfs = {}
    
    # First, load metadata for all satellite types
    for sat_type in satellite_types:
        metadata_path = os.path.join(subset_path, f"Core-{sat_type}", "metadata.parquet")
        if not os.path.exists(metadata_path):
            print(f"Skipping {sat_type}: metadata not found at {metadata_path}")
            continue
        
        gdf = load_metadata(metadata_path)
        local_dir = os.path.join(subset_path, f"Core-{sat_type}")
        
        # Split bands into tif and png based on configuration
        tif_bands = [b for b in bands_per_type[sat_type] if b in SATELLITE_CONFIGS[sat_type]['tif_bands']]
        png_bands = [b for b in bands_per_type[sat_type] if b in SATELLITE_CONFIGS[sat_type]['png_bands']]
        
        print(f"\nChecking files for {sat_type}...")
        
        # Check which indices have all required files
        valid_indices = []
        
        for idx in tqdm(range(len(gdf)), desc=f"Validating {sat_type} samples", unit="samples"):
            row = gdf.iloc[idx]
            grid_cell = row.grid_cell
            row_id = grid_cell.split('_')[0]
            product_id = row.product_id if 'product_id' in row.index else "id"
            
            base_path = os.path.join(local_dir, row_id, grid_cell, product_id)
            all_files_exist = True
            
            # Check TIF files
            for band in tif_bands:
                if not os.path.exists(os.path.join(base_path, f"{band}.tif")):
                    all_files_exist = False
                    break
            
            # Check PNG files
            if all_files_exist:  # Only check PNGs if TIFs exist
                for band in png_bands:
                    if not os.path.exists(os.path.join(base_path, f"{band}.png")):
                        all_files_exist = False
                        break
            
            if all_files_exist:
                valid_indices.append(idx)
        
        filtered_df = gdf.iloc[valid_indices].copy()
        print(f"Found {len(filtered_df)} valid samples out of {len(gdf)} for {sat_type}")
        filtered_dfs[sat_type] = filtered_df
    
    # Find common grid cells across all modalities
    grid_cell_sets = {
        source: set(df['grid_cell'].unique())
        for source, df in filtered_dfs.items()
    }
    
    # Find intersection of all grid cell sets
    common_grid_cells = set.intersection(*grid_cell_sets.values())
    print(f"\nFound {len(common_grid_cells)} common grid cells across all modalities")
    
    # Filter all modalities to keep only common grid cells
    for sat_type in satellite_types:
        if sat_type not in filtered_dfs:
            continue
            
        df = filtered_dfs[sat_type]
        df = df[df['grid_cell'].isin(common_grid_cells)]
        print(f"{sat_type}: {len(df)} samples for common grid cells")
        
        modalities[sat_type] = {
            'df': df,
            'local_dir': os.path.join(subset_path, f"Core-{sat_type}"),
            'tif_bands': tif_bands,
            'png_bands': png_bands,
            'tif_transforms': SATELLITE_CONFIGS[sat_type]['tif_transforms'],
            'png_transforms': SATELLITE_CONFIGS[sat_type]['png_transforms']
        }
    
    dataset = NMajorTOM(modalities=modalities, ratio_train_test=ratio_train_test, seed=seed)
    
    return dataset, len(common_grid_cells)

def is_valid_image(filepath):
    """Check if an image file is valid and can be opened. Deletes the file if corrupted."""
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            img.verify()  # Verify it's actually an image
        return True
    except Exception:
        print(f"  Warning: Corrupted or invalid image found: {filepath}")
        try:
            os.remove(filepath)
            print(f"  Deleted corrupted file: {filepath}")
        except Exception as e:
            print(f"  Failed to delete corrupted file {filepath}: {e}")
        return False

def get_existing_complete_grid_cells(output_dir, satellite_types, bands_per_type, num_grid_cells, expected_patches=16):
    """Returns a set of grid_cells that already have all their patches for all modalities"""
    complete_grid_cells_by_sat = {}
    corrupted_grid_cells = set()  # Track grid cells with corrupted files

    for sat_type in satellite_types:
        sat_base_dir = f"{sat_type}_{'_'.join(bands_per_type[sat_type])}"
        complete_grid_cells_by_sat[sat_type] = set()
        
        # Check both train and test directories
        for split in ['train', 'test']:
            dir_path = os.path.join(output_dir, split, sat_base_dir)
            print(f"  Checking {dir_path} for existing complete grid cells")
            if not os.path.exists(dir_path):
                print(f"  Warning: Directory {dir_path} does not exist")
                continue

            # Get all PNG files and extract their grid cells
            png_files = glob.glob(os.path.join(dir_path, "*.png"))
            print(f"  Found {len(png_files)} PNG files in {dir_path}")
            current_grid_cells = {}
            
            for f in png_files:
                # This will now delete the file if it's corrupted
                if not is_valid_image(f):
                    # Get the grid cell from the corrupted file
                    base_name = os.path.basename(f)
                    corrupted_grid_cell = "_".join(base_name.split("_")[:-1])
                    # Add to set of corrupted grid cells
                    corrupted_grid_cells.add(corrupted_grid_cell)
                    # Remove this grid cell from our complete cells since we'll need to regenerate it
                    if corrupted_grid_cell in current_grid_cells:
                        del current_grid_cells[corrupted_grid_cell]
                    continue
                    
                base_name = os.path.basename(f)
                grid_cell = "_".join(base_name.split("_")[:-1])  # Remove patch number
                current_grid_cells[grid_cell] = current_grid_cells.get(grid_cell, 0) + 1
            
            # Keep only grid cells with exactly the expected number of patches
            complete_cells = {gc for gc, count in current_grid_cells.items() if count == expected_patches}
            print(f"  Found {len(complete_cells)} complete grid cells in {split} split for {sat_type}")
            complete_grid_cells_by_sat[sat_type].update(complete_cells)
        
        print(f"Total complete grid cells for {sat_type}: {len(complete_grid_cells_by_sat[sat_type])}")

    # Find grid cells that are complete across all satellite types
    if not complete_grid_cells_by_sat:
        return set()
    
    complete_grid_cells = set.intersection(*complete_grid_cells_by_sat.values())
    
    # Remove any grid cells that had corrupted files
    complete_grid_cells = complete_grid_cells - corrupted_grid_cells
    
    # Print detailed debugging information
    print("\nComplete grid cells by satellite type:")
    for sat_type, cells in complete_grid_cells_by_sat.items():
        print(f"{sat_type}: {len(cells)} grid cells")
    print(f"\nGrid cells complete across all types: {len(complete_grid_cells)}")
    if corrupted_grid_cells:
        print(f"Removed {len(corrupted_grid_cells)} grid cells due to corrupted files")
    
    if len(complete_grid_cells) < num_grid_cells:
        # Find which grid cells are missing from which satellite types
        all_grid_cells = set.union(*complete_grid_cells_by_sat.values())
        print("\nAnalyzing missing grid cells:")
        for grid_cell in all_grid_cells:
            missing_from = [sat_type for sat_type in satellite_types 
                          if grid_cell not in complete_grid_cells_by_sat[sat_type]]
            if missing_from:
                print(f"Grid cell {grid_cell} is missing from: {', '.join(missing_from)}")

    return complete_grid_cells

def crop_images(dataset, satellite_types, bands_per_type, output_dir, num_grid_cells, flip=False, center_crop=False):
    """Extract features for all modalities simultaneously while ensuring they're paired"""
    from concurrent.futures import ThreadPoolExecutor
    import itertools
    
    # Create output directories if saving PNGs
    for sat_type in satellite_types:
        sat_base_dir = f"{sat_type}_{'_'.join(bands_per_type[sat_type])}"
        os.makedirs(os.path.join(output_dir, 'train', sat_base_dir), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', sat_base_dir), exist_ok=True)

    # Get already processed grid cells
    print("Checking for existing complete grid cells...")
    # Adjust the expected patch count based on center_crop mode
    expected_patches = 1 if center_crop else GRID_SIZE * GRID_SIZE
    # complete_grid_cells = get_existing_complete_grid_cells(output_dir, satellite_types, bands_per_type, num_grid_cells, expected_patches)
    complete_grid_cells = set()
    print(f"Found {len(complete_grid_cells)} already processed grid cells")

    # Pre-calculate patch positions (only used if not center_crop)
    patch_positions = list(itertools.product(range(GRID_SIZE), range(GRID_SIZE)))
    
    def process_sample(sample):
        """Process a single sample (large image) and return metadata for all its patches"""
        # Check if this grid cell is already processed
        grid_cell = sample[satellite_types[0]]['grid_cell']
        if grid_cell in complete_grid_cells:
            print(f"Skipping {grid_cell} because it already has all its patches")
            return []

        sample_metadata = []
        
        for sat_type in satellite_types:
            modality_data = sample[sat_type]
            split = modality_data['split']
            grid_cell = modality_data['grid_cell']
            
            img = modality_data['thumbnail']
            
            if center_crop:
                # Calculate center crop coordinates
                h, w = img.shape[-2:]
                start_h = (h - PATCH_SIZE) // 2
                start_w = (w - PATCH_SIZE) // 2
                patch = img[:, start_h:start_h + PATCH_SIZE, start_w:start_w + PATCH_SIZE]
                patches = patch.unsqueeze(0)  # Add batch dimension
            else:
                # Original patchifying logic
                C = img.size(0)
                patches = img.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
                patches = patches.permute(0, 1, 2, 3, 4).reshape(C, -1, PATCH_SIZE, PATCH_SIZE)
                patches = patches.permute(1, 0, 2, 3)  # [N_patches, C, H, W]
            
            if sat_type == 'DEM':
                patches = patches.repeat(1, 3, 1, 1)
            
            # Compute paths once
            sat_base_dir = f"{sat_type}_thumbnail"
            save_dir = os.path.join(output_dir, split, sat_base_dir)
            
            # Batch denormalize
            patches_denorm = (patches.detach().cpu() + 1) / 2
            
            # Save images
            for patch_idx, patch in enumerate(patches_denorm):
                if center_crop:
                    filename = f"{grid_cell}_center.png"
                    metadata = {
                        'grid_cell': grid_cell,
                        'satellite': sat_type,
                        'bands': 'thumbnail',
                        'split': split,
                        'patch_num': 0,
                        'patch_row': (GRID_SIZE - 1) // 2,
                        'patch_col': (GRID_SIZE - 1) // 2
                    }
                else:
                    filename = f"{grid_cell}_{patch_idx}.png"
                    metadata = {
                        'grid_cell': grid_cell,
                        'satellite': sat_type,
                        'bands': 'thumbnail',
                        'split': split,
                        'patch_num': patch_idx,
                        'patch_row': patch_positions[patch_idx][0],
                        'patch_col': patch_positions[patch_idx][1]
                    }
                
                torchvision.utils.save_image(patch, os.path.join(save_dir, filename))
                sample_metadata.append(metadata)
                
        return sample_metadata

    # Process samples in parallel
    all_metadata = []
    total_samples = len(dataset)
    
    print(f"Processing {total_samples} samples...")
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a list to store futures
        futures = []
        
        # Submit tasks with progress bar around the dataset iteration
        for sample in tqdm(dataset, total=total_samples, 
                         desc="Processing samples", 
                         unit="sample",
                         dynamic_ncols=True):
            future = executor.submit(process_sample, sample)
            futures.append(future)
        
        # Collect results
        for future in futures:
            metadata = future.result()
            if metadata:  # Only add metadata for newly processed samples
                all_metadata.extend(metadata)
    
    # Convert to DataFrame and split by train/test
    if all_metadata:  # Only process if we have new metadata
        df = pd.DataFrame(all_metadata)
        train_df = df[df['split'] == 'train'].drop('split', axis=1)
        test_df = df[df['split'] == 'test'].drop('split', axis=1)
        
        # Load existing metadata if it exists and append new data
        train_path = os.path.join(output_dir, 'train_metadata.parquet')
        test_path = os.path.join(output_dir, 'test_metadata.parquet')
        
        if os.path.exists(train_path):
            existing_train = pd.read_parquet(train_path)
            train_df = pd.concat([existing_train, train_df], ignore_index=True)
            # Deduplicate based on all columns
            train_df = train_df.drop_duplicates(subset=['grid_cell', 'satellite', 'patch_num'])
        
        if os.path.exists(test_path):
            existing_test = pd.read_parquet(test_path)
            test_df = pd.concat([existing_test, test_df], ignore_index=True)
            # Deduplicate based on all columns
            test_df = test_df.drop_duplicates(subset=['grid_cell', 'satellite', 'patch_num'])
        
        # Save metadata
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        
        print(f"Processed {len(all_metadata) // (16 * len(satellite_types))} new grid cells")
        print(f"Total metadata: {len(train_df)} training and {len(test_df)} testing samples")
    else:
        print("No new grid cells to process")

def visualize_patches(dataset, satellite_types, bands_per_type, output_dir):
    """Visualize the coverage of patches in a world map"""
    # Take the first satellite type since they're all paired
    sat_type = satellite_types[0]
    modality = dataset.modalities[sat_type]
    df = modality['df']
    
    # Lets split into train and test.
    # First, add the split column to the dataframe based on the grid_cell_to_split dictionary
    df['split'] = df['grid_cell'].map(dataset.grid_cell_to_split)
    
    # Create coverage map
    coverage_img_all = get_coveragemap(df)
    coverage_img_train = get_coveragemap(df[df['split'] == 'train'])
    coverage_img_test = get_coveragemap(df[df['split'] == 'test'])
    coverage_img_train_test = get_coveragemap(df[df['split'] == 'train'], df[df['split'] == 'test'])
    
    # Save the coverage map
    coverage_path_all = os.path.join(output_dir, 'coverage_map_all.png')
    coverage_path_train = os.path.join(output_dir, 'coverage_map_train.png')
    coverage_path_test = os.path.join(output_dir, 'coverage_map_test.png')
    coverage_path_train_test = os.path.join(output_dir, 'coverage_map_train_test.png')
    coverage_img_all.save(coverage_path_all, format='PNG')
    coverage_img_train.save(coverage_path_train, format='PNG')
    coverage_img_test.save(coverage_path_test, format='PNG')
    coverage_img_train_test.save(coverage_path_train_test, format='PNG')
    print(f"Saved coverage maps to {coverage_path_all}, {coverage_path_train}, {coverage_path_test} and {coverage_path_train_test}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from MajorTOM dataset')
    parser.add_argument('--subset_path', required=True, help='Path to the subset folder')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory')
    parser.add_argument('--bands', nargs='+', required=True, help='Bands to process (e.g., B1 B2 B3 DEM vv vh)')
    parser.add_argument('--ratio_train_test', type=float, default=0.95, help='Ratio of training to testing data')
    parser.add_argument('--flip', action='store_true', help='Flip the patches')
    parser.add_argument('--visualize', action='store_true', help='Visualize the patches in a world map')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--center_crop', action='store_true', help='Use center crop instead of patchifying')
    args = parser.parse_args()

    # Get subset name from path
    subset_name = Path(args.subset_path).name
    
    print("Flip is set to", args.flip)
    print("Seed is set to", args.seed)
    print("Subset path is", args.subset_path)
    print("Bands are", args.bands)
    print("Ratio train test is", args.ratio_train_test)
    print("Visualize is set to", args.visualize)
    print("Center crop is set to", args.center_crop)
    # Create the main output directory
    all_bands = '_'.join(sorted(args.bands))
    os.makedirs(args.output_dir, exist_ok=True)

    # Group bands by satellite type
    bands_per_type = {}
    satellite_types = []
    for sat_type, config in SATELLITE_CONFIGS.items():
        all_sat_bands = config['tif_bands'] + config['png_bands']
        sat_bands = [b for b in args.bands if b in all_sat_bands]
        if sat_bands:
            bands_per_type[sat_type] = sat_bands
            satellite_types.append(sat_type)

    if satellite_types:
        # Process all satellite types together
        dataset, num_grid_cells = process_satellite(args.subset_path, satellite_types, bands_per_type, args.ratio_train_test, args.seed)
        
        if args.visualize:
            print("==> Visualizing patches...")
            visualize_patches(dataset, satellite_types, bands_per_type, args.output_dir)
            print("==> Done visualizing patches! Exiting...")
            # exit()

        print("==> Cropping images...")
        crop_images(dataset, satellite_types, bands_per_type, args.output_dir, num_grid_cells, 
                    flip=args.flip, center_crop=args.center_crop)

if __name__ == "__main__":
    main()