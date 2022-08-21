import os
from fractions import Fraction
from typing import Callable, Optional

from .common import SRDataset, pil_loader


class RealSR(SRDataset):

    paths = {
        ('hr', 'train', Fraction(2)): os.path.join('Train', '2'),
        ('hr', 'train', Fraction(3)): os.path.join('Train', '3'),
        ('hr', 'train', Fraction(4)): os.path.join('Train', '4'),
        ('hr', 'test', Fraction(2)): os.path.join('Test', '2'),
        ('hr', 'test', Fraction(3)): os.path.join('Test', '3'),
        ('hr', 'test', Fraction(4)): os.path.join('Test', '4'),
        ('real', 'train', Fraction(2)): os.path.join('Train', '2'),
        ('real', 'train', Fraction(3)): os.path.join('Train', '3'),
        ('real', 'train', Fraction(4)): os.path.join('Train', '4'),
        ('real', 'test', Fraction(2)): os.path.join('Test', '2'),
        ('real', 'test', Fraction(3)): os.path.join('Test', '3'),
        ('real', 'test', Fraction(4)): os.path.join('Test', '4'),
    }

    cameras = ['Canon', 'Nikon']
    
    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'real',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        super(RealSR, self).__init__(root=os.path.join(root, 'RealSR'),
                                    scale=scale,
                                    downsample=downsample,
                                    phase=phase,
                                    image_loader=image_loader,
                                    transform=transform,
                                    download=False,
                                    preload=preload,
                                    return_file_name=return_file_name,
                                    return_meta=return_meta,
                                    synthetic_downsample=synthetic_downsample,
                                    )
        self.synthetic_downsample = False
    
    def _check_integrity(self):

        return True

    
    def _init_examples(self):

        files = [os.path.join(self.root, camera, self.paths[self.downsample, self.phase, self.scale], file)
                    for camera in self.cameras
                    for file in os.listdir(os.path.join(self.root, camera,
                                                        self.paths[self.downsample, self.phase, self.scale])
                                           )
                    if file.endswith('HR.png')
                    ]
        self.examples = list(files)

    def __getitem__(self, index: int):
        
        hr = self.image_loader(self.examples[index])
        lr = self.image_loader(self.examples[index].replace('HR.png', f'LR{self.scale}.png'))
          
        
        if self.transform is not None:
            hr, lr = self.transform([hr, lr])
            
        
        if self.return_file_name:
            return hr, lr, os.path.relpath(self.examples[index], self.root)
        else:
            return hr, lr