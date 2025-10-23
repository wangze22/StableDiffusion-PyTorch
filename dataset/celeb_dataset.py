import glob
import os
import random
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许读取被截断的图像，减少 OSError


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        self.idx_to_cls_map = {}
        self.cls_to_idx_map ={}
        
        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']
            
        self.images, self.texts, self.masks = self.load_images(im_path)
        if 'text' in self.condition_types:
            # Lazy cache so we only read a caption file the first time it is requested
            self._caption_cache = {}
        if not self.use_latents:
            self.image_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        texts = []
        masks = []
        caption_dir = os.path.join(im_path, 'celeba-caption')
        mask_dir = os.path.join(im_path, 'CelebAMask-HQ-mask')
        img_dir = os.path.join(im_path, 'CelebA-HQ-img')

        # Collect image file paths once. scandir is faster than repeated globbing and gives direct DirEntry objects.
        entries = []
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    entries.append(entry.path)

        # Sort numerically when filenames are numbers, otherwise fall back to lexicographic order.
        def _sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return int(stem)
            except ValueError:
                return stem

        entries.sort(key=_sort_key)
        
        if 'image' in self.condition_types:
            label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
            self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}
        
        for fname in tqdm(entries):
            ims.append(fname)

            if 'text' in self.condition_types:
                im_name = os.path.split(fname)[1].split('.')[0]
                caption_path = os.path.join(caption_dir, '{}.txt'.format(im_name))
                if not os.path.exists(caption_path):
                    raise FileNotFoundError(f'Caption file not found for image {fname}')
                texts.append(caption_path)

            if 'image' in self.condition_types:
                im_name = int(os.path.split(fname)[1].split('.')[0])
                masks.append(os.path.join(mask_dir, '{}.png'.format(im_name)))
        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        if 'image' in self.condition_types:
            assert len(masks) == len(ims), "Condition Type Image but could not find masks for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} masks'.format(len(masks)))
        print('Found {} captions'.format(len(texts)))
        return ims, texts, masks

    def get_mask(self, index):
        r"""
        Method to get the mask of WxH ...
        """
        try:
            with Image.open(self.masks[index]) as mask_im:
                mask_im = np.array(mask_im, dtype = np.int64)
            mask_im = torch.from_numpy(mask_im)
            mask_im = F.interpolate(
                mask_im.unsqueeze(0).unsqueeze(0).float(),
                size = (self.mask_h, self.mask_w),
                mode = 'nearest',
                ).squeeze().long()
            mask_im = mask_im.clamp(0, self.mask_channels)
            one_hot = F.one_hot(mask_im, num_classes = self.mask_channels + 1).movedim(-1, 0).float()
            mask = one_hot[1:]  # discard background channel
            return mask
        except (OSError, UnidentifiedImageError) as e:
            # 遇到坏的/不可读的 mask，返回全零（相当于只有背景），并给出轻量提示
            print(f"Warning: Skipping corrupted mask at {self.masks[index]} ({e})")
            return torch.zeros(self.mask_channels, self.mask_h, self.mask_w, dtype = torch.float32)

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            captions = self._get_captions(index)
            cond_inputs['text'] = random.sample(captions, k=1)[0]
        if 'image' in self.condition_types:
            mask = self.get_mask(index)
            cond_inputs['image'] = mask
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # 最多尝试 10 次重采样，避免坏图中断
            for _ in range(10):
                try:
                    with Image.open(self.images[index]) as im:
                        im_tensor = self.image_transform(im)
                    break
                except (OSError, UnidentifiedImageError) as e:
                    print(f"Warning: corrupted image {self.images[index]} ({e}); resampling...")
                    index = random.randint(0, len(self.images) - 1)
            else:
                # 多次失败后返回占位零图像
                print("Error: too many corrupted images encountered; returning zero image.")
                im_tensor = torch.zeros(self.im_channels, self.im_size, self.im_size, dtype=torch.float32)

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs

    def _get_captions(self, index):
        caption_path = self.texts[index]
        if caption_path not in self._caption_cache:
            with open(caption_path, 'r', encoding='utf-8') as f:
                captions_im = [line.strip() for line in f if line.strip()]
            self._caption_cache[caption_path] = captions_im
        return self._caption_cache[caption_path]
