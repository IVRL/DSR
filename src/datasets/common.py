import os
import tempfile
from fractions import Fraction
from typing import Any, Callable, List, Optional

from PIL import Image
import numpy as np
import cv2 as cv
import torchvision
import torch
from torchvision.datasets.vision import VisionDataset
from src.train.options import args
from src.models.metardn import input_matrix_wpn

def pil_loader(path: str) -> Image.Image:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        Image.Image: _description_
    """
    # with open(path, 'rb') as f:
    #     image = Image.open(f)
    #     return image.convert('RGB')
    return Image.open(path, 'r').convert('RGB')


def raw_loader(path: str) -> np.ndarray:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        np.ndarray: _description_
    """
    return cv.imread(path, cv.IMREAD_UNCHANGED)


class SRDataset(VisionDataset):
    """Base dataset.

    Args:
        VisionDataset (_type_): _description_
    """
    urls: list
    paths: dict
    extensions = ('JPG', 'png', 'PNG')
    synthetic_downsample=False
    
    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool=False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:

        super(SRDataset, self).__init__(root, transform=transform)

        self.scale = scale
        self.downsample = downsample
        self.phase = phase
        self.return_file_name = return_file_name
        self.return_alitutude = 'altitude' in args.arch

        if download:
            self._download()
            if not self._check_integrity():
                self._generate_lr()

        if not self._check_integrity():
            raise RuntimeError((f'Dataset {os.path.basename(self.root)} '
                                'not found or corrupted. You can use '
                                'download=True to download it.'))

        self.examples = []
        self._init_phase()
        self._init_examples()
        self.image_loader = image_loader
        
        self.synthetic_downsample = synthetic_downsample
        self.return_meta = return_meta
    
    def _init_phase(self):
        pass
            
    def _check_integrity(self):

        return all(map(lambda x: os.path.isdir(os.path.join(self.root, x)), self.paths.values()))

    def _generate_lr(self):

        if self.scale in [1, 2, 3, 4]:
            return

        numerator, denominator = self.scale.numerator, self.scale.denominator

        src = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        for phase in ['train', 'valid']:
            if not self.paths.get(('bicubic', phase, self.scale), None):
                continue
            os.makedirs(os.path.join(self.root, self.paths[(
                'bicubic', phase, self.scale)]), exist_ok=True)
            os.makedirs(os.path.join(self.root, self.paths[(
                'hr', phase, self.scale)]), exist_ok=True)
            os.system((f"""matlab -nodisplay -r "cd('{os.path.join(src, 'matlab')}'); """
                       f"""scale_numerator={numerator}; scale_denominator={denominator}; """
                       f"""folder_src='{os.path.join(self.root, self.paths[('hr', phase, 1)])}'; """
                        f"""folder_des_hr='{os.path.join(self.root, self.paths[('hr', phase, self.scale)])}'; """
                       f"""folder_des_lr='{os.path.join(self.root, self.paths[('bicubic', phase, self.scale)])}'; """
                       """bicubic_downsample ; exit" """))
        return

    def _download(self) -> None:
        """Download dataset."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        os.makedirs(self.root, exist_ok=True)

        for data in self.urls:
            filename = None
            md5sum = None
            if isinstance(data, str):
                url = data
            else:
                url = data[0]
                if len(data) > 1:
                    md5sum = data[1]
                if len(data) > 2:
                    filename = data[2]
            torchvision.datasets.utils.download_and_extract_archive(
                url, self.root, filename=filename, md5=md5sum)
            

    def _get_image_file_list(self, folder) -> List[str]:

        return sorted([os.path.join(root, file)
                       for root, _, files in os.walk(folder)
                       for file in files if file.endswith(self.extensions)])

    def _init_examples(self):

        lr_files = self._get_image_file_list(
            os.path.join(self.root,
                         self.paths[(self.downsample, self.phase, self.scale)]))
        hr_files = self._get_image_file_list(
            os.path.join(self.root,
                         self.paths[('hr', self.phase, self.scale)]))

        assert len(lr_files) == len(hr_files)

        self.examples = [(hr, lr) for lr, hr in zip(lr_files, hr_files)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> List[Any]:

        images = []
        for i, path in enumerate(self.examples[index]):
            if i > 0 and self.synthetic_downsample:
                image = self.image_loader(os.path.join(os.path.dirname(path), 'bicubic.png'))
            else:
                image = self.image_loader(path)
            images.append(image)

        if self.transform is not None:
            images = self.transform(images)
        
        # if self.return_alitutude:
        #     _, height, width = images[1].size()
        #     scale_coord_map, mask = input_matrix_wpn(height, width, self.scale, only_altitude='altitude' in args.arch)
        
        if self.return_alitutude and not self.return_file_name:
            return images[0], images[1], torch.tensor(1)
        elif self.return_alitutude and self.return_file_name:
            return images[0], images[1], torch.tensor(1), os.path.relpath(self.examples[index][0], self.root)
        elif not self.return_alitutude and self.return_file_name:
            return images[0], images[1], os.path.relpath(self.examples[index][0], self.root)
        else:
            return images[0], images[1]