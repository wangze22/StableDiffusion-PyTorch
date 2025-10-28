import glob
import os
import pickle
from typing import List, Sequence

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from models.vqvae import VQVAE
from config import celebhq_vqvae as cfg


def _resolve_cuda_devices() -> Sequence[int]:
    """
    Derive the list of locally visible CUDA device indices.
    Respects CUDA_VISIBLE_DEVICES and gracefully falls back to CPU.
    """
    if not torch.cuda.is_available():
        return ()

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return ()

    # torch has already remapped ids wrt CUDA_VISIBLE_DEVICES, so they are 0..count-1 here.
    return tuple(range(device_count))


CUDA_DEVICE_IDS: Sequence[int] = _resolve_cuda_devices()
PRIMARY_DEVICE = torch.device(f'cuda:{CUDA_DEVICE_IDS[0]}' if CUDA_DEVICE_IDS else 'cpu')

# Enable heuristics that help larger batch inference when a GPU backend is available.
if CUDA_DEVICE_IDS:
    torch.backends.cudnn.benchmark = True


def infer(
    task_name,
    num_samples,
    num_grid_rows,
    vqvae_checkpoint_name,
    save_latents,
    latent_dir_name,
    latent_batch_size = None,
    num_workers = None,
):
    ######## Load config from celebhq_vqvae module #######
    dataset_config = cfg.dataset_config
    autoencoder_config = cfg.autoencoder_model_config
    train_config = getattr(cfg, 'train_config', {})

    available_devices: Sequence[int] = CUDA_DEVICE_IDS
    device = PRIMARY_DEVICE

    # Create the dataset
    im_dataset_cls = {
        'mnist'  : MnistDataset,
        'celebhq': CelebDataset,
        }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        )

    num_images = num_samples
    ngrid = num_grid_rows

    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    model = VQVAE(
        im_channels = dataset_config['im_channels'],
        model_config = autoencoder_config,
        )
    map_location = 'cpu' if not available_devices else device
    state_dict = torch.load(vqvae_checkpoint_name, map_location = map_location)
    model.load_state_dict(state_dict)

    gpu_count = len(available_devices)
    if gpu_count:
        print(f'Visible CUDA devices: {list(available_devices)}')
    else:
        print('No CUDA devices detected. Falling back to CPU inference.')
    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids = list(available_devices))

    model = model.to(device)
    model.eval()

    with torch.no_grad():

        decoded_output, encoded_output, _ = model(ims)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2

        encoder_grid = make_grid(encoded_output.cpu(), nrow = ngrid)
        decoder_grid = make_grid(decoded_output.cpu(), nrow = ngrid)
        input_grid = make_grid(ims.cpu(), nrow = ngrid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)

        input_grid.save(os.path.join(task_name, 'input_samples.png'))
        encoder_grid.save(os.path.join(task_name, 'encoded_samples.png'))
        decoder_grid.save(os.path.join(task_name, 'reconstructed_samples.png'))

        if save_latents:
            # save latents with batched, order-preserving inference
            latent_path = latent_dir_name
            # latent_fnames = glob.glob(os.path.join(latent_dir_name, '*.pkl'))
            # assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))

            if latent_batch_size is None or latent_batch_size < 1:
                per_gpu_default = max(1, train_config.get('autoencoder_batch_size', 4))
                latent_batch_size = per_gpu_default * max(1, gpu_count)
            latent_batch_size = min(latent_batch_size, len(im_dataset))

            if num_workers is None:
                cpu_count = os.cpu_count() or 1
                # allocate workers proportionally to GPU count but capped by CPU availability
                num_workers = min(cpu_count, max(2, gpu_count * 2)) if gpu_count else 0

            data_loader = DataLoader(
                im_dataset,
                batch_size = latent_batch_size,
                shuffle = False,
                num_workers = num_workers,
                pin_memory = gpu_count > 0,
            )

            fname_latent_map = {}
            part_count = 0
            count = 0
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                _, latent_batch, _ = model(images.float().to(device))
                latent_batch = latent_batch.detach().cpu()

                batch_start = batch_idx * latent_batch_size
                for sample_offset in range(latent_batch.size(0)):
                    dataset_idx = batch_start + sample_offset
                    fname_latent_map[im_dataset.images[dataset_idx]] = latent_batch[sample_offset]
                    count += 1

                    # Save latents every 1000 images
                    if count % 1000 == 0:
                        pickle.dump(
                            fname_latent_map, open(f'{latent_path}/{part_count}.pkl', 'wb'),
                            )
                        part_count += 1
                        fname_latent_map = {}

            if len(fname_latent_map) > 0:
                pickle.dump(
                    fname_latent_map, open(f'{latent_path}/{part_count}.pkl', 'wb'),
                    )
            print('Done saving latents')


if __name__ == '__main__':
    # Configuration parameters
    task_name = 'model_pths'
    num_samples = 36
    num_grid_rows = 6
    vqvae_checkpoint_name = 'runs_VQVAE_noise_server/vqvae_20251028-022443_save/celebhq/n_scale_0.1000/vqvae_autoencoder_ckpt_latest.pth'
    save_latents = True
    latent_dir_name = 'vqvae_latents_28'

    infer(task_name, num_samples, num_grid_rows, vqvae_checkpoint_name, save_latents, latent_dir_name)
