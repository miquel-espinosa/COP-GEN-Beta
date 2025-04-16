"""
Author: Chenhongyi Yang
Reference: GPViT https://github.com/ChenhongyiYang/GPViT
"""

"""
This script will generate a paired LMDB database for all modalities found in the input directory.
Thus, the input directory should contain subdirectories for each modality, each containing a set of images.
The names for the paired images in the different subdirectories should be the same.

# Example:
python3 scripts/create_lmdb.py \
    --input-img-dir data/majorTOM/northern_italy/northern_italy_thumbnail_npy/train \
    --output-dir data/majorTOM/northern_italy/northern_italy_thumbnail_npy_lmdb/train \
    --input-type npy
"""


import glob
import blobfile as bf
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import pickle

import cv2
import lmdb

import argparse
parser = argparse.ArgumentParser('Convert LMDB dataset')
parser.add_argument('--input-img-dir', help='Path to ImageNet training images')
parser.add_argument('--output-dir', help='Path to output training lmdb dataset')
parser.add_argument('--input-type', choices=['png', 'npy'],
                    help='Type of input to encode: "png" for PNG images or "npy" for NPY features')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Batch size for processing images')
# parser.add_argument('val-img-dir', 'Path to ImageNet validation images')
# parser.add_argument('val-out', 'Path to output validation lmdb dataset')
args = parser.parse_args()

_10TB = 10 * (1 << 40)

class LmdbDataExporter(object):
    """
    making LMDB database
    """
    # label_pattern = re.compile(r'/.*/.*?(\d+)$')

    def __init__(self,
                 img_dir=None,
                 output_path=None,
                 batch_size=None):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.img_dir = img_dir
        self.output_path = output_path
        self.batch_size = batch_size

        if not os.path.exists(img_dir):
            raise Exception(f'{img_dir} does not exist!')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)
        self.modalities = self._get_modalities()

    def _get_modalities(self):
        """Get list of modalities (subdirectories) in the input directory"""
        return [d for d in os.listdir(self.img_dir) 
                if os.path.isdir(os.path.join(self.img_dir, d))]

    def export(self):
        idx = 0
        results = []
        st = time.time()
        iter_img_lst = self.read_imgs()
        length = self.get_length()
        print(f'length: {length}')
        while True:
            items = []
            try:
                while len(items) < self.batch_size:
                    items.append(next(iter_img_lst))
            except StopIteration:
                break

            with ThreadPoolExecutor() as executor:
                results.extend(executor.map(self._extract_once, items))

            if len(results) >= self.batch_size:
                self.save_to_lmdb(results)
                idx += self.batch_size
                et = time.time()
                print(f'time: {(et-st)}(s)  count: {idx}')
                st = time.time()
                # Progressively decrease batch size for remaining items
                remaining = length - idx
                if remaining < self.batch_size:
                    self.batch_size = max(remaining // 2, 1)
                    print(f'batch_size is reduced to: {self.batch_size}')
                del results[:]

        et = time.time()
        print(f'time: {(et-st)}(s)  count: {idx}')
        self.save_to_lmdb(results)
        # self.save_total(idx)
        print('Total length:', len(results))
        del results[:]

    def save_to_lmdb(self, results):
        """
        persist to lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(img_key, img_byte)

    def save_total(self, total: int):
        """
        persist all numbers of imgs
        """
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(total).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        image_name = item[1]
        modality_data = item[2]  # Dictionary of modality -> file path
        
        # Create a dictionary to store all modality data
        data_dict = {}
        
        # Read each modality's data
        for modality, file_path in modality_data.items():
            if args.input_type == 'image':
                img = cv2.imread(file_path)
                if img is None:
                    print(f'{file_path} is a bad img file.')
                    return None, None
                _, img_byte = cv2.imencode('.png', img)
                data_dict[modality] = img_byte.tobytes()
            else:  # feature
                try:
                    import numpy as np
                    features = np.load(file_path)
                    data_dict[modality] = features.tobytes()
                except Exception as e:
                    print(f'Error loading {file_path}: {e}')
                    return None, None
        
        return (image_name.encode('ascii'), pickle.dumps(data_dict))

    def get_length(self):
        # Just count files in the first modality directory
        if not self.modalities:
            return 0
        first_modality_dir = os.path.join(self.img_dir, self.modalities[0])
        img_list = glob.glob(os.path.join(first_modality_dir, '*.npy'))
        return len(img_list)
    
    def _list_image_files_recursively(self, data_dir):
        results = []
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(self._list_image_files_recursively(full_path))
        return results

    def read_imgs(self):
        # Create a dictionary to store files by their base name
        file_groups = defaultdict(dict)
        
        # File extension based on input type
        extensions = ['.png'] if args.input_type == 'png' else ['.npy']
        
        # Collect files from each modality
        for modality in self.modalities:
            modality_path = os.path.join(self.img_dir, modality)
            for file_path in self._list_image_files_recursively(modality_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in extensions:
                    base_name = os.path.basename(file_path)
                    file_groups[base_name][modality] = file_path
        
        # Only yield complete groups
        for idx, (base_name, modality_files) in enumerate(file_groups.items()):
            if len(modality_files) == len(self.modalities):
                item = (idx, base_name, modality_files)
                yield item
            else:
                print(f"Skipping incomplete group {base_name}, found modalities: {list(modality_files.keys())}")


if __name__ == '__main__':
    input_img_dir = args.input_img_dir
    output_dir = args.output_dir

    exporter = LmdbDataExporter(
        input_img_dir,
        output_dir,
        batch_size=args.batch_size)
    exporter.export()
