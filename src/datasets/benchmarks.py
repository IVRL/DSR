import os
import tempfile
import shutil
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union
from tenacity import RetryAction

import torchvision
from .common import SRDataset, pil_loader


class Set5(SRDataset):
    """`Set5 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    urls = [
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    paths = {
        ('hr', 'test', 1) : os.path.join('HR'),
        ('hr', 'test', 2) : os.path.join('HR'),
        ('hr', 'test', 3) : os.path.join('HR'),
        ('hr', 'test', 4) : os.path.join('HR'),
        ('hr', 'test', Fraction(50, 9)) : os.path.join('HR_X50_9'),
        ('bicubic', 'test', 2) : os.path.join('LR_bicubic', 'X2'),
        ('bicubic', 'test', 3) : os.path.join('LR_bicubic', 'X3'),
        ('bicubic', 'test', 4) : os.path.join('LR_bicubic', 'X4'),
        ('bicubic', 'test', Fraction(50, 9)): os.path.join('LR_bicubic', 'X50_9'),
    }
    
    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        super(Set5, self).__init__(root=os.path.join(root, 'Set5'),
                                    scale=scale,
                                    downsample=downsample,
                                    phase=phase,
                                    image_loader=image_loader,
                                    transform=transform,
                                    download=download,
                                    preload=preload,
                                    return_file_name=return_file_name,
                                    return_meta=return_meta,
                                    synthetic_downsample=synthetic_downsample,
                                    )
        self.synthetic_downsample = False
    
    def _download(self) -> None:
        """Download dataset."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        with tempfile.TemporaryDirectory() as temp:
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
                    url, temp, filename=filename, md5=md5sum)
            shutil.move(os.path.join(temp, 'benchmark', 'Set5'), self.root)



class Set14(SRDataset):
    """`Set14 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    urls = [
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    paths = {
        ('hr', 'test', 1) : os.path.join('HR'),
        ('hr', 'test', 2) : os.path.join('HR'),
        ('hr', 'test', 3) : os.path.join('HR'),
        ('hr', 'test', 4) : os.path.join('HR'),
        ('hr', 'test', Fraction(50, 9)) : os.path.join('HR_X50_9'),
        ('bicubic', 'test', 2) : os.path.join('LR_bicubic', 'X2'),
        ('bicubic', 'test', 3) : os.path.join('LR_bicubic', 'X3'),
        ('bicubic', 'test', 4) : os.path.join('LR_bicubic', 'X4'),
        ('bicubic', 'test', Fraction(50, 9)): os.path.join('LR_bicubic', 'X50_9'),
    }

    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        super(Set14, self).__init__(root=os.path.join(root, 'Set14'),
                                    scale=scale,
                                    downsample=downsample,
                                    phase=phase,
                                    image_loader=image_loader,
                                    transform=transform,
                                    download=download,
                                    preload=preload,
                                    return_file_name=return_file_name,
                                    return_meta=return_meta,
                                    synthetic_downsample=synthetic_downsample,
                                    )
        self.synthetic_downsample = False

    def _download(self) -> None:
        """Download dataset."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        with tempfile.TemporaryDirectory() as temp:
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
                    url, temp, filename=filename, md5=md5sum)
            shutil.move(os.path.join(temp, 'benchmark', 'Set14'), self.root)


class B100(SRDataset):
    """`B100 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    urls = [
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    paths = {
        ('hr', 'test', 1) : os.path.join('HR'),
        ('hr', 'test', 2) : os.path.join('HR'),
        ('hr', 'test', 3) : os.path.join('HR'),
        ('hr', 'test', 4) : os.path.join('HR'),
        ('hr', 'test', Fraction(50, 9)) : os.path.join('HR_X50_9'),
        ('bicubic', 'test', 2) : os.path.join('LR_bicubic', 'X2'),
        ('bicubic', 'test', 3) : os.path.join('LR_bicubic', 'X3'),
        ('bicubic', 'test', 4) : os.path.join('LR_bicubic', 'X4'),
        ('bicubic', 'test', Fraction(50, 9)): os.path.join('LR_bicubic', 'X50_9'),
    }

    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        super(B100, self).__init__(root=os.path.join(root, 'B100'),
                                    scale=scale,
                                    downsample=downsample,
                                    phase=phase,
                                    image_loader=image_loader,
                                    transform=transform,
                                    download=download,
                                    preload=preload,
                                    return_file_name=return_file_name,
                                    return_meta=return_meta,
                                    synthetic_downsample=synthetic_downsample,
                                    )
        self.synthetic_downsample = False
    
    def _download(self) -> None:
        """Download dataset."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        with tempfile.TemporaryDirectory() as temp:
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
                    url, temp, filename=filename, md5=md5sum)
            shutil.move(os.path.join(temp, 'benchmark', 'B100'), self.root)


class Urban100(SRDataset):
    """`Urban100 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    urls = [
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    paths = {
        ('hr', 'test', 1) : os.path.join('HR'),
        ('hr', 'test', 2) : os.path.join('HR'),
        ('hr', 'test', 3) : os.path.join('HR'),
        ('hr', 'test', 4) : os.path.join('HR'),
        ('hr', 'test', Fraction(50, 9)) : os.path.join('HR_X50_9'),
        ('bicubic', 'test', 2) : os.path.join('LR_bicubic', 'X2'),
        ('bicubic', 'test', 3) : os.path.join('LR_bicubic', 'X3'),
        ('bicubic', 'test', 4) : os.path.join('LR_bicubic', 'X4'),
        ('bicubic', 'test', Fraction(50, 9)): os.path.join('LR_bicubic', 'X50_9'),
    }

    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        super(Urban100, self).__init__(root=os.path.join(root, 'Urban100'),
                                    scale=scale,
                                    downsample=downsample,
                                    phase=phase,
                                    image_loader=image_loader,
                                    transform=transform,
                                    download=download,
                                    preload=preload,
                                    return_file_name=return_file_name,
                                    return_meta=return_meta,
                                    synthetic_downsample=synthetic_downsample,
                                    )
        self.synthetic_downsample = False
    
    def _download(self) -> None:
        """Download dataset."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        with tempfile.TemporaryDirectory() as temp:
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
                    url, temp, filename=filename, md5=md5sum)
            shutil.move(os.path.join(temp, 'benchmark', 'Urban100'), self.root)
        
        
