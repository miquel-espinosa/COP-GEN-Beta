import torch.nn as nn
import numpy as np
import torch
from datasets import MajorTOMThumbnail
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
import os

torch.manual_seed(0)
np.random.seed(0)


def get_existing_encoded_files(output_dir, image_paths):
    """Returns a set of filenames that already have their encoded features saved"""
    existing_files = set()
    missing_files = set()
    
    for img_path in image_paths:
        filename = os.path.basename(img_path).split('.')[0]
        npy_path = os.path.join(output_dir, f'{filename}.npy')
        
        if os.path.exists(npy_path):
            existing_files.add(filename)
        else:
            missing_files.add(filename)
    
    print(f"\nFound {len(existing_files)} already encoded files")
    print(f"Missing {len(missing_files)} files to encode")
    
    return existing_files, missing_files

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    datafactory = MajorTOMThumbnail(path=args.path, resolution=args.resolution)
    dataset = datafactory.get_split(split=None, labeled=False, nosplit=True)
    image_paths = dataset.image_paths
    
    # Check for existing encoded files
    # existing_files, missing_files = get_existing_encoded_files(args.output_dir, image_paths)
    # TODO: Restart is not working yet
    existing_files = set()
    missing_files = set(image_paths)
    
    if len(missing_files) == 0:
        print("All files have already been encoded. Exiting...")
        return
        
    # Filter dataset to only process missing files
    filtered_indices = [i for i, path in enumerate(image_paths) 
                       if os.path.basename(path).split('.')[0] not in existing_files]
    dataset.image_paths = [image_paths[i] for i in filtered_indices]
    
    dataset_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    # model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    processed_count = 0
    for img, filename in tqdm(dataset_loader, desc="Encoding images", unit="batch"):
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        for moment, fname in zip(moments, filename):
            np.save(f'{args.output_dir}/{fname}.npy', moment)
            processed_count += 1

    print(f'\nProcessed {processed_count} new files')
    print(f'Total encoded files: {len(existing_files) + processed_count}')

    # features = []
    # labels = []
    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print(f'features.shape={features.shape}')
    # print(f'labels.shape={labels.shape}')
    # np.save(f'imagenet{resolution}_features.npy', features)
    # np.save(f'imagenet{resolution}_labels.npy', labels)


if __name__ == "__main__":
    main()
