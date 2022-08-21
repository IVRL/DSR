import os
import json
import random
from fractions import Fraction

from typing import Callable, Optional
import torch

from .common import SRDataset, pil_loader
from src.train.options import args
from src.models.metardn import input_matrix_wpn

class DroneSRJPG(SRDataset):
    
    # heights = [10, 20, 30, 40, 50, 70, 80, 100, 120, 140]
    
    def __init__(self,
                 root: str,
                 scale: Fraction = 1,
                 downsample: str = 'bicubic',
                 phase: str = 'train',
                 image_loader: Callable = pil_loader,
                 transform: Optional[Callable] = None,
                 valid_ratio: float = 0.1,
                 heights: list = [10],
                 download: bool = False,
                 preload: bool = False,
                 return_file_name: bool=False,
                 return_meta: bool=False,
                 synthetic_downsample: bool=False,
                 ) -> None:
        
        assert scale == Fraction(50, 9)
        self.heights = heights
        self.valid_ratio = valid_ratio
        self.paths = dict()   
        super(DroneSRJPG, self).__init__(root=os.path.join(root, 'DroneSR-X7'),
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
        
    def _init_phase(self):
    
        split_file = os.path.join(self.root, 'train_valid_test_split.json')
        if os.path.isfile(split_file):
            with open(split_file, 'r') as file:
                self.split = json.load(file)
        else:
            self.split = {'train': [], 'valid': [], 'test': []}
            all = [x for x in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, x))]
            if len(self.split['test']) < 20:
                self.split['test'] = random.choices(all, k=20)
            all = [x for x in all if x not in self.split['test']]
            if len(self.split['valid']) < 20:
                self.split['valid'] = random.choices(all, k=20)
            all = [x for x in all if x not in self.split['valid']]
            self.split['train'] = all
            with open(split_file, 'w') as file:
                json.dump(self.split, file)
    
    def _init_examples(self):

        if args.meta:
            self.examples = []
            for height in self.heights:
                file_per_height = [os.path.join(self.root, scene, str(height), patch) 
                                   for scene in self.split[self.phase] if os.path.isdir(os.path.join(self.root, scene, str(height)))
                                   for patch in os.listdir(os.path.join(self.root, scene, str(height))) 
                                   if os.path.isdir(os.path.join(self.root, scene, str(height), patch))]
                random.shuffle(file_per_height)
                self.examples.append(file_per_height)
        else:
            files = [(os.path.join(self.root, scene, str(height), patch), int(height))
                        for scene in self.split[self.phase]
                        for height in self.heights if os.path.isdir(os.path.join(self.root, scene, str(height)))
                        for patch in os.listdir(os.path.join(self.root, scene, str(height))) 
                        if os.path.isdir(os.path.join(self.root, scene, str(height), patch))]
            self.examples = list(files)
    
    def __len__(self) -> int:
        if args.meta:
            return min(map(len, self.examples))
        else:
            return len(self.examples)
    
    def __getitem__(self, index: int):
        
        if args.meta:
            hrs = []
            lrs = []
            altitudes = []
            files = []
            for i, altitude in enumerate(self.heights):
                folder = self.examples[i][index]
                hr = self.image_loader(os.path.join(folder, 'tele.png'))
                if args.color_correction:
                    lr = self.image_loader(os.path.join(folder, 'color_correction.png'))
                else:
                    lr = self.image_loader(os.path.join(folder, 'hasselblad0.png'))
                if self.transform is not None:
                    hr, lr = self.transform([hr, lr])
                if not args.send_altitude:
                    altitude = 1
                hrs.append(hr)
                lrs.append(lr)
                altitudes.append(torch.tensor(altitude))
                files.append(os.path.relpath(folder, self.root))
            if len(hrs) == 1:
                if self.return_file_name:
                    return hrs[0], lrs[0], altitudes[0], files[0]
                else:
                    return hrs[0], lrs[0], altitudes[0]
            else:
                return torch.stack(hrs), torch.stack(lrs), torch.stack(altitudes)
                
        else:
            folder, altitude = self.examples[index]
            hr = self.image_loader(os.path.join(folder, 'tele.png'))
            if self.synthetic_downsample:
                lr = self.image_loader(os.path.join(folder, 'bicubic.png'))
            else:
                if args.color_correction:
                    lr = self.image_loader(os.path.join(folder, 'color_correction.png'))
                else:
                    lr = self.image_loader(os.path.join(folder, 'hasselblad0.png'))
        
            if self.transform is not None:
                hr, lr = self.transform([hr, lr])
            
            if not args.send_altitude:
                altitude = 80
                
            if self.return_meta and not self.return_file_name:
                return hr, lr, torch.tensor(altitude) / 80
            elif self.return_meta and self.return_file_name:
                return hr, lr, torch.tensor(altitude) / 80 , os.path.relpath(folder, self.root)
            elif not self.return_meta and self.return_file_name:
                return hr, lr, os.path.relpath(folder, self.root)
            else:
                return hr, lr
    
    def shuffle(self):
        if args.meta:
            for file_per_height in self.examples:
                random.shuffle(file_per_height)
            
        
        