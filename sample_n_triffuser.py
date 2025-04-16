import ml_collections
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import time
import copy
from datasets import get_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.utils._pytree import tree_map
import glob
import os
import functools
from concurrent.futures import ThreadPoolExecutor

# Add profiling tools
class Profiler:
    def __init__(self):
        self.times = {}
        
    def profile(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            func_name = func.__name__
            if func_name not in self.times:
                self.times[func_name] = []
            self.times[func_name].append(end_time - start_time)
            
            return result
        return wrapper
    
    def summary(self):
        print("\n----- Profiling Summary -----")
        for func_name, times in self.times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            calls = len(times)
            print(f"{func_name}: {total_time:.2f}s total, {avg_time:.4f}s avg, {calls} calls")
        print("----------------------------\n")

profiler = Profiler()

MODALITIES = {
    4: ['dem', 's1_rtc', 's2_l1c', 's2_l2a'],
    3: ['dem', 's1_rtc', 's2_l1c'],
    2: ['s1_rtc', 's2_l2a'],
}

MODEL_RESOLUTION = 256

"""
Sampling for any n modalities

> python3 sample_n_triffuser.py --config=path --data_path=path --nnet_path=path \
                    --n_mod=int --n_samples=int \
                    --generate=[modalities] --condition=[modalities]


Generate all modalities unconditional (joint):
python3 sample_n_triffuser.py --n_mod=4 --generate=s2_l1c,s2_l2a,s1_rtc,dem

Generate a pair unconditional (joint):
python3 sample_n_triffuser.py --n_mod=4 --generate=s1_rtc,s2_l2a

Generate s1_rtc and s2_l2a, conditioned on dem and s2_l1c (conditional):
python3 sample_n_triffuser.py --n_mod=4 --generate=s1_rtc,s2_l2a --condition=dem,s2_l1c

Generate dem conditioned on s1_rtc, s2_l1c, s2_l2a (conditional):
python3 sample_n_triffuser.py --n_mod=4 --generate=dem --condition=s1_rtc,s2_l1c,s2_l2a

Generate dem conditioned on s1_rtc (conditional) (the rest are automatically ignored: s2_l1c, s2_l2a):
python3 sample_n_triffuser.py --n_mod=4 --generate=dem --condition=s1_rtc 

Generate dem unconditional (marginal) (no condition, the rest are ignored):
python3 sample_n_triffuser.py --n_mod=4 --generate=dem


Note:
--generate flag is mandatory
"generate" modalities and "condition" modalities should always be different


"""

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = glob.glob(os.path.join(folder_path, "*.png"))
        print("There are", len(self.image_files), "images in the dataset")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return both the image and the filename
        return image, os.path.basename(self.image_files[idx])



def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

@profiler.profile
def prepare_contexts(config, images, filenames, device, autoencoder=None):
    """
    If a modality is conditional, we need to return the npy feature encodings
    If a modality is unconditional, we need to return random noise
    
    batch_shape = (n_modalities, B, C, H, W)
    
    Returns:
        img_contexts: Tensor containing contexts for each modality
        processed_filenames: List of filenames, duplicated and labeled with version suffixes if n_samples > 1
    """
    
    # Create a noise tensor with the same shape as the images batch
    if config.data_type == 'lmdb':
        effective_batch_size = images[0].shape[0] * config.n_samples if config.n_samples > 1 else images[0].shape[0]
        img_contexts = torch.randn(config.num_modalities, effective_batch_size, *images[0].shape[1:], device=device)
    elif config.data_type == 'folder-img':
        # Calculate effective batch size (original batch size * n_samples)
        effective_batch_size = images.shape[0] * config.n_samples if config.n_samples > 1 else images.shape[0]
        # Multiply the images_batch shape by 2 because we have both mean and variance
        # as output from the autoencoder
        img_contexts = torch.randn(config.num_modalities, effective_batch_size, 2 * config.z_shape[0],
                                   config.z_shape[1], config.z_shape[2], device=device)

    # Process filenames - duplicate them if n_samples > 1 and add version suffixes
    processed_filenames = []
    if config.n_samples > 1:
        for filename in filenames:
            for i in range(config.n_samples):
                processed_filenames.append(f"{filename}_v{i+1}")
    else:
        processed_filenames = filenames
        
    # For each modality in the images_batch, if it is conditional, load and duplicate the npy feature encodings
    for i, modality in enumerate(config.modalities):
        if config.condition_modalities_mask[i]:
            if config.data_type == 'lmdb':
                # Duplicate each conditional input n_samples times
                img_contexts[i] = images[i].repeat_interleave(config.n_samples, dim=0)
            elif config.data_type == 'folder-img':
                assert autoencoder is not None, "Autoencoder must be provided for folder-img data type"
                # Duplicate each conditional input n_samples times
                duplicated_batch = images.repeat_interleave(config.n_samples, dim=0)
                img_contexts[i] = autoencoder.encode_moments(duplicated_batch)
                
                # Padding the latents experiment
                # duplicated_batch = images.repeat_interleave(config.n_samples, dim=0)
                # intermediate_latents = autoencoder.encode_moments(duplicated_batch)
                # padded_latents = torch.nn.functional.pad(intermediate_latents, (8, 8, 8, 8), mode='reflect')
                # img_contexts[i] = padded_latents

    return img_contexts, processed_filenames

def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    
def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Create output directory once at the start
    os.makedirs(config.output_path, exist_ok=True)
    # Create a directory for each modality if we are saving as pngs
    if config.save_as == 'pngs':
        for modality in config.generate_modalities:
            os.makedirs(os.path.join(config.output_path, modality), exist_ok=True)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.seed)

    config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    nnet = utils.get_nnet(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()
    
    if config.data_type == 'lmdb':
        # Edit the dataset path to the data path from the command line arguments
        dataset_config = ml_collections.ConfigDict(config.to_dict())
        dataset_config.dataset.path = config.data_path
        
        # Always return the filename
        dataset_config.dataset.return_filename = True
            
        dataset = get_dataset(**dataset_config.dataset)
        # TODO: This is not intuitive. Split is train but it is returning the test set. See datasets.py
        test_dataset = dataset.get_split(split='train', labeled=False)
        # Create a generator with fixed seed for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(config.seed)  # Using the same seed as set earlier in the code
        dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
                               num_workers=8, pin_memory=True, persistent_workers=True, generator=g)

    elif config.data_type == 'folder-img':
        print("config.data_path", config.data_path)
        
        if config.resolution >= MODEL_RESOLUTION:
            transform = standard_transforms.Compose([
                standard_transforms.CenterCrop(MODEL_RESOLUTION),
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        else:
            padding_4sides = (MODEL_RESOLUTION - config.resolution) // 2
            transform = standard_transforms.Compose([
                standard_transforms.CenterCrop(config.resolution),
                standard_transforms.ToTensor(),
                torch.nn.ReflectionPad2d(padding_4sides),
                standard_transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        dataset = CustomImageDataset(config.data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False,
                               num_workers=8, pin_memory=True, persistent_workers=True)
    else:
        raise ValueError(f"Invalid data type: {config.data_type}. Must be one of ['lmdb', 'folder-img']")
    
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
   
   
    @profiler.profile
    def split_joint(x, z_imgs, config):
        """
        Input:
        x:      (B, C, H, W) 
                is only the modalities that are being denoised 
        z_imgs: (M, B, C, H, W)
                the original img_latents for all modalities
                (but we only use the ones for the modalities that are being denoised) 
        config: config
        
        First, split the input into the modalities into correct shape
        Second, return a full list of the modalities,
        including the ones being conditioned on and the ones being ignored.
        
        Returns list of all modalities (some are denoised, some are conditioned on, some are ignored)
        
        """
        
        C, H, W = config.z_shape
        z_dim = C * H * W
        z_generated = x.split([z_dim] * len(config.generate_modalities), dim=1)
        z_generated = {modality: einops.rearrange(z_i, 'B (C H W) -> B C H W', C=C, H=H, W=W)
                       for z_i, modality in zip(z_generated, config.generate_modalities)}
        
        z = []
        for i, modality in enumerate(config.modalities):
            # Modalities that are being denoised
            if modality in config.generate_modalities:
                z.append(z_generated[modality])
            # Modalities that are being conditioned on
            elif modality in config.condition_modalities:
                z.append(z_imgs[i])
            # Modalities that are ignored
            else:
                z.append(torch.randn(x.shape[0], C, H, W, device=device))
        
        return z


    @profiler.profile
    def combine_joint(z):
        """
        Input:
        z: list of ONLY the modalities that are being denoised
        Returns:
        z: (B, C * H * W)
        """
        z = torch.concat([einops.rearrange(z_i, 'B C H W -> B (C H W)') for z_i in z], dim=-1)
        return z
    
    @torch.cuda.amp.autocast()
    @profiler.profile
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    @profiler.profile
    def decode(_batch):
        return autoencoder.decode(_batch)
    
    def get_data_generator():
        # Run single epoch
        for data in tqdm(dataloader, desc='epoch'):
            yield data
    
    logging.info("Num of modalities: %d", config.num_modalities)
    logging.info("Num of images in dataloader: %d", len(dataloader))
    logging.info("Generate modalities: %s", config.generate_modalities)
    logging.info("Condition modalities: %s", config.condition_modalities)
    logging.info("Condition modalities mask: %s", config.condition_modalities_mask)
    logging.info("Generate modalities mask: %s", config.generate_modalities_mask)
    logging.info(f'N={N}')
    
    
    @profiler.profile
    def run_nnet(x, t, z_imgs):
        
        timesteps = [t if mask else torch.zeros_like(t) for mask in config.generate_modalities_mask]
        
        # ==== EXPAND TO ALL MODALITIES ====
        z = split_joint(x, z_imgs, config=config)
        # z = {modality1: z_generated_modality1, modality2: z_conditioned_modality2, ...}
        
        # == DEBUG CODE: Decode, unprocess, and save both modalities side by side
        # z_decoded_1 = decode(z[0])
        # z_decoded_2 = decode(z[1])
        # z_decoded_1 = unpreprocess(z_decoded_1)
        # z_decoded_2 = unpreprocess(z_decoded_2)
        # z_decoded_combined = torch.cat([z_decoded_1, z_decoded_2], dim=-1)  # Concatenate along width dimension
        # print(f"saving image z_decoded_combined_{t}.png")
        # save_image(z_decoded_combined, os.path.join(config.output_path, f"z_supeeerdecoded_{t}.png"))
        # == DEBUG CODE END ==
        
        """
        nnet expects:
         - z: (M, B, C, H, W)
         - t_imgs: (M, B)
        where M is the number of modalities.
        
        That is, z should be a list of M batches, each batch corresponding to a modality.
        E.g. num_modalities(M)=3, batch_size(B)=16, z_shape(C, H, W)=(4, 32, 32) ->
           z = [(16, 4, 32, 32), (16, 4, 32, 32), (16, 4, 32, 32)]
           t_imgs = [(16,), (16,), (16,)]
        """
        
        z_out = nnet(z, t_imgs=timesteps)
        
        # ==== SELECT ONLY THE GENERATED MODALITIES for the denoising process ====
        z_out_generated = [z_out[i]
                            for i, modality in enumerate(config.modalities)
                                if modality in config.generate_modalities]
        
        x_out = combine_joint(z_out_generated)
        
        if config.sample.scale == 0.:
            return x_out
        
        return x_out  # TODO: Implement classifier-free guidance if there is time
    
    @profiler.profile
    def sample_fn(z_imgs, **kwargs):
        # Calculate effective batch size
        effective_batch_size = z_imgs[0].shape[0]
        
        # Generate random initial noise for the modalities being generated/denoised
        _z_init = torch.randn(len(config.generate_modalities), effective_batch_size, *z_imgs[0].shape[1:], device=device)
        
        _x_init = combine_joint(_z_init)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
        
        @profiler.profile
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return run_nnet(x, t, z_imgs)
        
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {config.batch_size} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

        _zs = split_joint(x, z_imgs, config=config)
        
        # Replace the conditional modalities with the original images
        for i, mask in enumerate(config.condition_modalities_mask):
            if mask:
                _zs[i] = z_imgs[i]
                
        return _zs
    
    
    data_generator = get_data_generator()
    for idx_batch, batch in enumerate(data_generator):

        batch_start_time = time.time()
        
        # Unpack the batch into images and filenames
        original_images, original_filenames = batch
        
        # print(filenames)
        
        # Track data loading and preprocessing time
        preprocess_start = time.time()
        images = tree_map(lambda x: x.to(device), original_images)
        # In addition to preparing the contexts (returns mean and variance),
        # we need to actually sample the values from the distribution
        img_contexts, filenames = prepare_contexts(config, images, original_filenames, device=device, autoencoder=autoencoder)
        z_imgs = torch.stack([autoencoder.sample(img_context) for img_context in img_contexts])
        preprocess_time = time.time() - preprocess_start
        
        # Track sampling time
        sample_start = time.time()
        _zs = sample_fn(z_imgs)
        sample_time = time.time() - sample_start
        
        # Track decoding time
        decode_start = time.time()
        samples_unstacked = [unpreprocess(decode(_z)) for _z in _zs]
        
        # Crop back to input resolution if it is smaller than MODEL_RESOLUTION
        if config.resolution < MODEL_RESOLUTION:
            samples_unstacked = [standard_transforms.functional.center_crop(sample, output_size=config.resolution)
                                    for sample in samples_unstacked]
        
        samples = torch.stack(samples_unstacked, dim=0)
        decode_time = time.time() - decode_start
        
        # Track saving time
        save_start = time.time()
        
        if config.save_as == 'grid':
        
            b = samples.shape[1]  # batch size
            # Properly interleave samples from all modalities
            # For each sample index, get all modalities before moving to next sample
            samples = torch.stack([samples[j, i] for i in range(b) for j in range(config.nnet.num_modalities)]).view(-1, *samples.shape[2:])
            # If the number of modalities is 3 then we plot in 9 columns
            n_cols = 9 if config.nnet.num_modalities == 3 else 8
            samples = make_grid(samples, n_cols)
            save_path = os.path.join(config.output_path, f'grid_{idx_batch}.png')
            save_image(samples, save_path)
            

            # plot_real_images = '/home/s2254242/projects/pangaea_terramind/data/test_set_1/test' # We want to plot into a grid_real_images_{idx_batch}.png the real images
            plot_real_images = ''
            
            if plot_real_images != '':
                # Load real images from files
                real_images_list = []
                for filename in original_filenames:
                    for modality in config.modalities:
                        img_path = os.path.join(plot_real_images, modality, f"{filename}.png")
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = standard_transforms.ToTensor()(img)
                        real_images_list.append(img_tensor)
                
                # Stack and create grid
                real_images = torch.stack(real_images_list)
                real_grid = make_grid(real_images, n_cols)
                real_save_path = os.path.join(config.output_path, f'grid_real_{idx_batch}.png')
                save_image(real_grid, real_save_path)
            
        elif config.save_as == 'pngs':
            # Define a helper function to save a single image
            def save_single_image(args):
                modality_idx, modality, b_idx = args
                filename = filenames[b_idx] if isinstance(filenames, list) else filenames
                save_path = os.path.join(os.path.join(config.output_path, modality), f"{filename}.png")
                save_image(samples[modality_idx][b_idx], save_path)
            
            # Create a list of all save operations needed
            save_tasks = []
            for i, modality in enumerate(config.modalities):
                if modality in config.generate_modalities:
                    modality_dir = os.path.join(config.output_path, modality)
                    for b_idx in range(samples[i].shape[0]):
                        save_tasks.append((i, modality, b_idx))
            
            # Use ThreadPoolExecutor to parallelize the saving process
            max_workers = min(16, len(save_tasks))  # Limit to 16 threads max
            if max_workers > 0:  # Only create pool if there are tasks
                print(f"Saving {len(save_tasks)} images using {max_workers} threads...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(tqdm(executor.map(save_single_image, save_tasks), total=len(save_tasks), desc="Saving images"))
        
        elif config.data_type == 'folder-img':
            # Get indices for all modalities we want to save
            save_modalities = ['s1_rtc', 's2_l2a']
            # append_real_from_paths = ['data/pastis_pngs/sar/']
            modality_indices = [config.modalities.index(m) for m in save_modalities]
            
            for i in range(min(config.batch_size, len(filenames))):
                # Stack the samples from different modalities horizontally
                concat_samples = torch.cat([samples[idx, i] for idx in modality_indices], dim=2)
                
                # Append real images from specified paths
                real_images = []
                for real_path in append_real_from_paths:
                    real_img_path = os.path.join(real_path, filenames[i])
                    real_img = Image.open(real_img_path).convert("RGB")
                    real_img_tensor = standard_transforms.ToTensor()(real_img)
                    real_images.append(real_img_tensor)
                
                real_images_tensor = torch.cat(real_images, dim=2) if len(real_images) > 1 else real_images[0]
                concat_samples = torch.cat([concat_samples, real_images_tensor.to(device)], dim=2)
                
                save_path = os.path.join(config.output_path, filenames[i])
                save_image(concat_samples, save_path)
        
        save_time = time.time() - save_start
        
        batch_total_time = time.time() - batch_start_time
        
        print(f'\nBatch {idx_batch} timing:')
        print(f'  Preprocessing: {preprocess_time:.2f}s ({preprocess_time/batch_total_time*100:.1f}%)')
        print(f'  Sampling:      {sample_time:.2f}s ({sample_time/batch_total_time*100:.1f}%)')
        print(f'  Decoding:      {decode_time:.2f}s ({decode_time/batch_total_time*100:.1f}%)')
        print(f'  Saving:        {save_time:.2f}s ({save_time/batch_total_time*100:.1f}%)')
        print(f'  Total:         {batch_total_time:.2f}s')

        print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        print(f'\nresults are saved in {os.path.join(config.output_path)} :)')
        
        # After processing, display the profiling summary
        if idx_batch % 5 == 0 or idx_batch == len(dataloader) - 1:
            profiler.summary()


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Configuration.", lock_config=False)
flags.DEFINE_string("data_path", None, "Path to the data")
flags.DEFINE_string("data_type", 'lmdb', "Type of data to load (lmdb, folder-img)")
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to save the generated images")
flags.DEFINE_integer("n_mod", None, "Number of modalities")
flags.DEFINE_integer("n_samples", 1, "The number of samples to generate with the same condition")
flags.DEFINE_string("generate", None, "Comma-separated list of modalities to generate (s2_l1c,s2_l2a,s1_rtc,dem)")
flags.DEFINE_string("condition", None, "Comma-separated list of modalities to condition on (s2_l1c,s2_l2a,s1_rtc,dem)")
flags.DEFINE_string("save_as", 'grid', "How to save the generated images (grid, pngs)")
flags.DEFINE_integer("resolution", 256, "The resolution of the images to generate")
flags.DEFINE_integer("seed", None, "Random seed for reproducibility (overrides config seed)")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.data_path = FLAGS.data_path
    config.save_as = FLAGS.save_as
    config.n_samples = FLAGS.n_samples if FLAGS.n_samples else 1
    config.resolution = FLAGS.resolution
    
    # Override seed if provided from command line
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    
    # batch_size controls the number of unique conditional images we use
    config.batch_size = 6
    
    config.modalities = MODALITIES[FLAGS.n_mod]
    
    if FLAGS.generate is None:
        raise ValueError("--generate flag is mandatory")
    
    # Parse generate and condition modalities
    config.generate_modalities = FLAGS.generate.split(',')
    config.condition_modalities = FLAGS.condition.split(',') if FLAGS.condition else []
    
    # Sort the modalities by the order of the config.modalities
    config.generate_modalities = sorted(config.generate_modalities, key=lambda x: config.modalities.index(x))
    config.condition_modalities = sorted(config.condition_modalities, key=lambda x: config.modalities.index(x))
    
    config.generate_modalities_mask = [mod in config.generate_modalities for mod in config.modalities]
    config.condition_modalities_mask = [mod in config.condition_modalities for mod in config.modalities]
    
    # Validate modalities
    valid_modalities = {'s2_l1c', 's2_l2a', 's1_rtc', 'dem'}
    for mod in config.generate_modalities + config.condition_modalities:
        if mod not in valid_modalities:
            raise ValueError(f"Invalid modality: {mod}. Must be one of {valid_modalities}")
    
    # Check that generate and condition modalities don't overlap
    if set(config.generate_modalities) & set(config.condition_modalities):
        raise ValueError("Generate and condition modalities must be different")
    
    if FLAGS.data_type == 'lmdb':
        # Check that there exists a data.mdb and a lock.mdb in the data path
        if not os.path.exists(os.path.join(config.data_path, 'data.mdb')):
            raise ValueError(f"data.mdb does not exist in {config.data_path}")
        if not os.path.exists(os.path.join(config.data_path, 'lock.mdb')):
            raise ValueError(f"lock.mdb does not exist in {config.data_path}")
    elif FLAGS.data_type == 'folder-img':
        # raise NotImplementedError("Folder-img data type not implemented")
        pass
    else:
        raise ValueError(f"Invalid data type: {FLAGS.data_type}. Must be one of ['lmdb', 'folder-img']")
    config.data_type = FLAGS.data_type
    
    assert config.nnet.num_modalities == FLAGS.n_mod, "Number of modalities in the nnet must match the number of modalities in the command line arguments"
    config.num_modalities = FLAGS.n_mod
    
    # Format the output path based on conditions and modalities
    clean_generate = [mod.replace('_', '') for mod in config.generate_modalities]
    if config.condition_modalities:
        clean_condition = [mod.replace('_', '') for mod in config.condition_modalities]
        output_dir = f"condition_{'_'.join(clean_condition)}_generate_{'_'.join(clean_generate)}_{config.n_samples}samples"
    else:
        output_dir = f"generate_{'_'.join(clean_generate)}_{config.n_samples}samples"
        
    if config.save_as == 'grid':
        config.output_path = os.path.join(FLAGS.output_path, 'grids', output_dir)
    else:
        config.output_path = os.path.join(FLAGS.output_path, output_dir)

    evaluate(config)
    
    # Print final profiling summary
    print("\n===== FINAL PROFILING SUMMARY =====")
    profiler.summary()


if __name__ == "__main__":
    app.run(main)
