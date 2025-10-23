import glob
import os
import pickle

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from models.vqvae import VQVAE
from config import celebhq_vqvae as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(task_name, num_samples, num_grid_rows, vqvae_checkpoint_name, save_latents, latent_dir_name):
    ######## Load config from celebhq_vqvae module #######
    dataset_config = cfg.dataset_config
    autoencoder_config = cfg.autoencoder_model_config

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

    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(
        im_dataset,
        batch_size = 1,
        shuffle = False,
        )

    num_images = num_samples
    ngrid = num_grid_rows

    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    model = VQVAE(
        im_channels = dataset_config['im_channels'],
        model_config = autoencoder_config,
        ).to(device)
    model.load_state_dict(torch.load(vqvae_checkpoint_name, map_location = device))
    model.eval()

    with torch.no_grad():

        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
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
            # save Latents (but in a very unoptimized way)
            latent_path = latent_dir_name
            # latent_fnames = glob.glob(os.path.join(latent_dir_name, '*.pkl'))
            # assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))

            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, im in enumerate(tqdm(data_loader)):
                encoded_output, _ = model.encode(im.float().to(device))
                fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                if (count + 1) % 1000 == 0:
                    pickle.dump(
                        fname_latent_map, open(f'{latent_path}/{part_count}.pkl', 'wb'),
                        )
                    part_count += 1
                    fname_latent_map = {}
                count += 1
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
    vqvae_checkpoint_name = 'model_pths/vqvae_autoencoder_ckpt_latest.pth'
    save_latents = True
    latent_dir_name = 'vqvae_latents_22'

    infer(task_name, num_samples, num_grid_rows, vqvae_checkpoint_name, save_latents, latent_dir_name)
