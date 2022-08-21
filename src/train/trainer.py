import copy
from curses.ascii import alt
import os
import tempfile
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
import piq
from tqdm.auto import tqdm
from PIL import Image
import mlflow
from mlflow.tracking import MlflowClient

from .enums import *

from .helpers import AverageMeter
from .helpers import to_image
from .helpers import to_tensor
from .helpers import to_luminance
from .helpers import to_YCbCr
from .helpers import get_datasets
from .helpers import get_model
from .helpers import get_optimizer
from .helpers import get_scheduler
from .helpers import get_loss
from .helpers import get_device
from .helpers import get_dtype

from .options import args
from src.meta import MAML

class Trainer:
    def __init__(self):
        self.iteration = 0
        self.epoch = 0
        self.best_psnr = None
        self.best_ssim = None
        self.best_loss = None
        self.best_epoch = None
        
        self.setup_device()
        self.setup_datasets()
        self.setup_model()
        self.load_from_mlflow()
        self.distribute_model()
        self.setup_meta() 
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()
        self.load_checkpoint()
        self.setup_tensorboard()
    
    def setup_meta(self):
        if args.meta:
            print('Using MAML wapper')
            self.model = MAML(self.model, lr=args.maml_lr, first_order=args.fomaml)
            
    def setup_device(self):
        self.device = get_device()
        self.dtype = get_dtype()

    def setup_datasets(self):
        self.loader_train, self.loader_val = get_datasets()

    def setup_model(self):
        self.model = get_model()
            
    def distribute_model(self):
        if self.device == torch.device('cuda') and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device).to(self.dtype)

    def setup_optimizer(self):
        self.optimizer = get_optimizer(self.model)

    def setup_scheduler(self):
        
        self.scheduler = None
        if args.meta:
            self.min_learning_rate = 9e-6
            self.learning_rate_change_iter_nums = [0]
            self.loss_steps = []
            self.loss_values = []
            self.learning_rate = args.lr
        else:
            self.scheduler = get_scheduler(self.optimizer)

    def setup_loss(self):
        self.loss_fn = get_loss()

    def setup_tensorboard(self):
        self.writer = None
        if not args.validation_only:
            try:
                # Only if tensorboard is present
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(args.log_dir, purge_step=self.epoch)
            except ImportError:
                if args.log_dir is not None:
                    raise ImportError("tensorboard is required to use --log-dir")

    def train_iter(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat), leave=False, position=1)
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            for i in range(args.dataset_repeat):
                for data in self.loader_train:
                    
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    
                    if 'altitude' in args.arch:
                        hr, lr, altitude = data
                        hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                        altitude = altitude.to(self.dtype).to(self.device)
                        if 'simplenet' in args.arch:
                            lr = F.interpolate(lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                        sr = self.model(lr, altitude)
                        sr = self.process_for_eval(sr)
                        hr = self.process_for_eval(hr)
                        loss = self.loss_fn(sr, hr)
                    else:
                        hr, lr = data
                        hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                        if 'simplenet' in args.arch:
                            lr = F.interpolate(lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                        sr = self.model(lr)
                        sr = self.process_for_eval(sr)
                        hr = self.process_for_eval(hr)
                        loss = self.loss_fn(sr, hr)
                    
                    loss.backward()
                    if args.gradient_clipping is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                    self.optimizer.step()
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        'L1': f'{l1_loss:.4f}',
                        'L2': f'{l2_loss:.4f}'
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)

            if args.report_to_mlflow:
                mlflow.log_metrics({'Train-L1': float(l1_avg.get()),
                                    'Train-L2': float(l2_avg.get()),
                                    }, step=self.epoch)

    def val_iter(self, final=True):
        
        # run = []
        with torch.no_grad():
            self.model.eval()
            if final:
                t = tqdm(self.loader_val, position=0)
                t.set_description("Validation")
            else:
                t = tqdm(self.loader_val, leave=False, position=1)
                t.set_description(f"Epoch {self.epoch} val ")
                
            psnr_avg = 0
            ssim_avg = 0
           
            l1_avg = 0
            l2_avg = 0
            loss_avg = 0
            if final:
                gmsd_avg = 0
                lpips_loss = piq.LPIPS()
                lpips_avg = 0
            cnt = 0
            
            for data in t:
    
                if 'altitude' in args.arch:
                    hr, lr, altitude, file = data
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    altitude = altitude.to(self.dtype).to(self.device)
                    if 'simplenet' in args.arch:
                        lr = F.interpolate(lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                    sr = self.model(lr, altitude)
                else:
                    hr, lr, file = data
                    if 'simplenet' in args.arch:
                        lr = F.interpolate(lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    sr = self.model(lr)
                n = sr.shape[0]
                if args.post_resize:
                    sr = F.interpolate(sr, scale_factor=25/18, mode='bicubic', align_corners=False)
                sr = sr.clamp(0, 1)
                if final:
                    # Round to pixel values
                    sr = sr.mul(255).round().div(255)
                
                sr = self.process_for_eval(sr)
                hr = self.process_for_eval(hr)
                loss = self.loss_fn(sr, hr)
                l1_loss = nn.functional.l1_loss(sr, hr).item()
                l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                psnr = piq.psnr(hr, sr, data_range=1.0)
                ssim = piq.ssim(hr, sr)
                loss_avg += loss.item() * n
                l1_avg += l1_loss * n
                l2_avg += l2_loss * n
                psnr_avg += psnr * n
                ssim_avg += ssim * n
                cnt += n
                args_dic = {
                    'PSNR': f'{psnr:.4f}',
                    'SSIM': f'{ssim:.4f}',
                }
                if final:
                    gmsd = piq.gmsd(hr, sr)
                    gmsd_avg += gmsd * n
                    lpips = lpips_loss(hr, sr)
                    lpips_avg += lpips.item() * n
                    args_dic.update({
                        'LPIPS': f'{lpips:.4f}',
                        'GMSD': f'{gmsd:.4f}',
                        })
                if args.batch_size_val == 1:
                    args_dic['Image'] =  file[0]
                    
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    args_dic[args.loss.name] = f'{loss.item():.4f}'
                t.set_postfix(**args_dic)
            
            loss_avg /= cnt
            l1_avg /= cnt
            l2_avg /= cnt
            psnr_avg /= cnt
            ssim_avg /= cnt
            if final:
                gmsd_avg /= cnt
                lpips_avg /= cnt
              
            if self.writer is not None:
                self.writer.add_scalar('PSNR', psnr_avg, self.epoch)
                self.writer.add_scalar('SSIM', ssim_avg, self.epoch)
                self.writer.add_scalar('L1', l1_avg, self.epoch)
                self.writer.add_scalar('L2', l2_avg, self.epoch)
                if final:
                    self.writer.add_scalar('GMSD', gmsd_avg, self.epoch)
                    self.writer.add_scalar('LPIPS', lpips_avg, self.epoch)
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    self.writer.add_scalar(args.loss.name, loss_avg, self.epoch)
            if args.report_to_mlflow:
                mlflow.log_metrics({'Valid-PSNR': float(psnr_avg),
                                    'Valid-SSIM': float(ssim_avg),
                                    'Valid-L1': float(l1_avg),
                                    'Valid-L2': float(l2_avg),
                                    }, step=self.epoch)
                if final:
                    mlflow.log_metrics({'Valid-LPIPS': float(lpips_avg),
                                        'Valid-GMSD': float(gmsd_avg),
                                        }, step=self.epoch)
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    mlflow.log_metric(key=f'Valid-{args.loss.value.upper()}',
                                      value=float(loss_avg),
                                      step=self.epoch)
            if final:
                return loss_avg, psnr_avg, ssim_avg, lpips_avg, gmsd_avg,
            else:
                return loss_avg, psnr_avg, ssim_avg

    def validation(self):
        if args.meta:
            loss, loss_std, psnr, psnr_std, ssim, ssim_std, lpips, lpips_std, gmsd, gmsd_std = self.val_meta(repeat=args.dataset_repeat)
            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM, LossType.LPIPS]:
                print(f"PSNR: {psnr:.2f}±{psnr_std}, SSIM: {ssim:.4f}±{ssim_std}, LPIPS: {lpips:.4f}±{lpips_std}, GMSD: {gmsd:.4f}±{gmsd_std}, {args.loss.value.upper()}: {loss:.4f}±{loss_std}")
            else:
                print(f"PSNR: {psnr:.2f}±{psnr_std}, SSIM: {ssim:.4f}±{ssim_std}, LPIPS: {lpips:.4f}±{lpips_std}, GMSD: {gmsd:.4f}±{gmsd_std}")
        else:
            loss, psnr, ssim, lpips, gmsd = self.val_iter()
            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM, LossType.LPIPS]:
                print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}, GMSD: {gmsd:.4f}, {args.loss.value.upper()}: {loss:.4f}")
            else:
                print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}, GMSD: {gmsd:.4f}")

    def run_model(self):
        scale = args.scale
        with torch.no_grad():
            self.model.eval()
            input_images = []
            for f in args.images:
                if os.path.isdir(f):
                    for g in os.listdir(f):
                        n = os.path.join(f, g)
                        if os.path.isfile(n):
                            input_images.append(n)
                else:
                    input_images.append(f)
            if args.destination is None:
                raise ValueError("You should specify a destination directory")
            os.makedirs(args.destination, exist_ok=True)
            t = tqdm(input_images)
            t.set_description("Run")
            for filename in t:
                try:
                    img = Image.open(filename)
                    img.load()
                except:
                    print(f"Could not open {filename}")
                    continue
                img = to_tensor(img).to(self.device)
                sr_img = self.model(img)
                sr_img = to_image(sr_img)
                destname = os.path.splitext(os.path.basename(filename))[0] + f"_x{scale}.png"
                sr_img.save(os.path.join(args.destination, destname))

    def train(self):
        t = tqdm(total=args.epochs, initial=self.epoch, position=0)
        t.set_description("Epochs")
        if self.best_epoch is not None:
            args_dic = {'best': self.best_epoch}
            if self.best_psnr is not None:
                args_dic['PSNR'] = f'{self.best_psnr:.2f}'
            if self.best_ssim is not None:
                args_dic['SSIM'] = f'{self.best_ssim:.2f}'    
            if self.best_loss is not None:
                args_dic['loss'] = f'{self.best_loss:.2f}'
            t.set_postfix(**args_dic)
        while self.epoch < args.epochs:
            self.epoch += 1
            self.train_iter()
            loss, psnr, ssim = self.val_iter(final=False)
            is_best = self.best_loss is None or loss < self.best_loss
            if is_best:
                self.best_loss = loss
                self.best_psnr = psnr
                self.best_ssim = ssim
                self.best_epoch = self.epoch
                t.set_postfix(best=self.epoch, PSNR=f'{psnr:.2f}',
                              SSIM=f'{ssim:.4f}', loss=f'{loss:.4f}')
            self.save_checkpoint(best=is_best)
            t.update(1)
            self.scheduler.step()

    def get_model_state_dict(self):
        # Ensures that the state_dict is on the CPU and reverse model transformations
        self.model.to('cpu')
        model = copy.deepcopy(self.model)
        self.model.to(self.device)
        if args.weight_norm:
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    m = nn.utils.remove_weight_norm(m)
        return model.state_dict()
    
    def load_from_mlflow(self):
        if args.load_from_mlflow:
            print("Load from mlflow")
            with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp:
                client =  MlflowClient() 
                client.download_artifacts(args.load_from_mlflow, 'model', temp)
                ckp = torch.load(os.path.join(temp, 'model', 'model_best.pth'), map_location=self.device)
                try:
                    self.model.load_state_dict(ckp['state_dict'], strict=True)
                except RuntimeError:
                    self.model.load_state_dict(ckp['state_dict'], strict=False)
                    print("Not perfect load!")
            #     if self.optimizer is not None:
            #         self.optimizer.load_state_dict(ckp['optimizer'])
            #     if self.scheduler is not None:
            #         self.scheduler.load_state_dict(ckp['scheduler'])
            #     self.epoch = ckp['epoch']
            #     if 'best_epoch' in ckp:
            #         self.best_epoch = ckp['best_epoch']
            #     if 'best_psnr' in ckp:
            #         self.best_psnr = ckp['best_psnr']
            #     if 'best_ssim' in ckp:
            #         self.best_ssim = ckp['best_ssim']
            #     if 'best_loss' in ckp:
            #         self.best_loss = ckp['best_loss']
        return


    def load_checkpoint(self):
        if args.load_checkpoint is None:
            return
        ckp = torch.load(args.load_checkpoint)
        self.model.load_state_dict(ckp['state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckp['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckp['scheduler'])
        self.epoch = ckp['epoch']
        if 'best_epoch' in ckp:
            self.best_epoch = ckp['best_epoch']
        if 'best_psnr' in ckp:
            self.best_psnr = ckp['best_psnr']
        if 'best_ssim' in ckp:
            self.best_ssim = ckp['best_ssim']
        if 'best_loss' in ckp:
            self.best_loss = ckp['best_loss']

    def save_checkpoint(self, best=False,):
        if args.save_checkpoint is None:
            return
        path = args.save_checkpoint
        if args.meta:
            state = {
                'state_dict': self.model.module.module.state_dict() if torch.cuda.device_count() > 1 else self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }
        else:
            state = {
                'state_dict': self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'best_epoch': self.best_epoch,
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_loss': self.best_loss,
            }
        torch.save(state, path)
        base, ext = os.path.splitext(path)
        if args.save_every is not None and self.epoch % args.save_every == 0:
            torch.save(state, base + f"_e{self.epoch}" + ext)
        if best:
            torch.save(state, base + "_best" + ext)
            # torch.save(self.get_model_state_dict(), base + "_model" + ext)

    def process_for_eval(self, img):
        if args.shave_border != 0:
            shave = args.shave_border
            img = img[..., shave:-shave, shave:-shave]
        if args.eval_luminance:
            img = to_luminance(img)
        elif args.scale_chroma is not None:
            img = to_YCbCr(img)
            chroma_scaling = torch.tensor([1.0, args.scale_chroma, args.scale_chroma])
            img = img * chroma_scaling.reshape(1, 3, 1, 1).to(img.device)
        return img
    
    
    def train_meta(self):
            
        t = tqdm(total=args.iterations, initial=self.iteration, position=0)
        t.set_description("Iterations")
        done = False
        self.model.train()
        while self.iteration < args.iterations:
            for hrs, lrs, altitudes in self.loader_train:
                if hrs.shape[0] < 2*args.shots:
                    continue
                self.iteration += 1
                iteration_loss = 0.0
                for i in range(hrs.shape[1]):
                    learner = self.model.clone()
                    hr = hrs[:, i, :, :, :]
                    lr = lrs[:, i, :, :, :]
                    altitude = altitudes[:, i]

                    hr, lr, altitude = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device), altitude.to(self.dtype).to(self.device)
                    
                    # Separate data into adaptation/evaluation sets
                    adaptation_indices = np.zeros(hr.size(0), dtype=bool)
                    adaptation_indices[np.arange(args.shots) * 2] = True
                    evaluation_indices = torch.from_numpy(~adaptation_indices).to(self.device)
                    adaptation_indices = torch.from_numpy(adaptation_indices).to(self.device)
                    adaptation_hr, adaptation_lr, adaptation_altitude = hr[adaptation_indices], lr[adaptation_indices], altitude[adaptation_indices]
                    evaluation_hr, evaluation_lr, evaluation_altitude = hr[evaluation_indices], lr[evaluation_indices], altitude[evaluation_indices]
                    
                    # Fast Adaptation
                    if 'simplenet' in args.arch:
                        adaptation_lr = F.interpolate(adaptation_lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                    for step in range(args.fas):
                        if 'altitude' in args.arch:
                            adaptation_sr = learner(adaptation_lr, adaptation_altitude)
                        else:
                            adaptation_sr = learner(adaptation_lr)
                        adaptation_sr = self.process_for_eval(adaptation_sr)
                        adaptation_hr = self.process_for_eval(adaptation_hr)
                        adaption_loss = self.loss_fn(adaptation_sr, adaptation_hr)
                        learner.adapt(adaption_loss)
                        
                    # Compute validation loss
                    if 'simplenet' in args.arch:
                        evaluation_lr = F.interpolate(evaluation_lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                    if 'altitude' in args.arch:
                        evaluation_sr = learner(evaluation_lr, evaluation_altitude)
                    else:
                        evaluation_sr = learner(evaluation_lr)
                    evaluation_sr = self.process_for_eval(evaluation_sr)
                    evaluation_hr = self.process_for_eval(evaluation_hr)
                    evaluation_loss = self.loss_fn(evaluation_sr, evaluation_hr)
                    iteration_loss += evaluation_loss

                iteration_loss /= hrs.shape[1]
                # Take the meta-learning step
                self.optimizer.zero_grad()
                iteration_loss.backward()
                self.optimizer.step()
                args_dic = {args.loss.name: f'{iteration_loss.item():.4f}'}
                t.update()
                t.set_postfix(**args_dic)
                if args.report_to_mlflow:
                    mlflow.log_metrics({'Train-Loss': float(iteration_loss.item()),
                                        }, step=self.iteration)
                
                if self.iteration % args.run_test_every == 0:
                    loss, psnr, ssim = self.val_meta(final=False)
                    self.loss_steps.append(self.iteration)
                    self.loss_values.append(loss)
                    self.learning_rate_policy()
                    if self.learning_rate < args.min_learning_rate:
                        done = True
                if self.iteration > args.iterations:
                    done = True
                if done:
                    break
            if done:
                break
            self.loader_train.dataset.shuffle()
        self.save_checkpoint(best=True)
    
    def val_meta(self, final=True, repeat:int=1):
        
        if final:
            t = tqdm(total=sum(map(len, self.loader_val)) * repeat, position=0)
            t.set_description("Validation")
        else:
            t = tqdm(total=sum(map(len, self.loader_val)) * repeat, leave=False, position=1)
            t.set_description(f"Iteration {self.iteration} val ")

        psnr_history = []
        ssim_history = []
        l1_history = []
        l2_history = []
        loss_history = []
        if final:
            lpips_loss = piq.LPIPS()
            gmsd_history = []
            lpips_history = []
        
        for _ in range(repeat):
            for altitude, dataloader in zip(args.val_altitudes, self.loader_val):
                for shot, (hr, lr, altitude, file) in enumerate(dataloader):
                    
                    if shot % (2 * args.shots) == 0:
                        learner = self.model.clone()
                        
                    if (shot % (2 * args.shots)) < args.shots and args.validation_only:
                        hr = center_crop(hr, args.patch_size_train)
                        lr = center_crop(lr, args.patch_size_train / args.scale)
                    hr, lr, altitude = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device), altitude.to(self.dtype).to(self.device)
                    if 'simplenet' in args.arch:
                        lr = F.interpolate(lr, scale_factor=args.scale, mode='bicubic', align_corners=False)
                    
                    if (shot % (2 * args.shots)) < args.shots:
                        # start = time.time()
                        for step in range(args.fas):
                            
                            if 'altitude' in args.arch:
                                sr = learner(lr, altitude)
                            else:
                                
                                sr = learner(lr)
                            sr = self.process_for_eval(sr)
                            hr = self.process_for_eval(hr)
                            loss = self.loss_fn(sr, hr)
                            learner.adapt(loss)
                    else:
                        if (shot % (2 * args.shots)) == args.shots:
                            learner.eval() 
                        with torch.no_grad():
                            if 'altitude' in args.arch:
                                sr = learner(lr, altitude)
                            else:
                                sr = learner(lr)
                            sr = sr.clamp(0, 1)
                            if final:
                                # Round to pixel values
                                sr = sr.mul(255).round().div(255)
                            
                            sr = self.process_for_eval(sr)
                            hr = self.process_for_eval(hr)
                            loss = self.loss_fn(sr, hr).item()
                            l1_loss = nn.functional.l1_loss(sr, hr).item()
                            l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                            psnr = piq.psnr(hr, sr, data_range=1.0)
                            ssim = piq.ssim(hr, sr)
                            
                            loss_history.append(loss)
                            l1_history.append(l1_loss)
                            l2_history.append(l2_loss)
                            psnr_history.append(psnr.item())
                            ssim_history.append(ssim.item())
                            
                            args_dic = {
                                'PSNR': f'{psnr:.4f}',
                                'SSIM': f'{ssim:.4f}',
                                
                                'Image': file[0],
                            }
                            
                            if final:
                                gmsd = piq.gmsd(hr, sr).item()
                                lpips = lpips_loss(hr, sr).item()
                                gmsd_history.append(gmsd)
                                lpips_history.append(lpips)
                                args_dic.update({
                                    'LPIPS': f'{lpips:.4f}',
                                    'GMSD': f'{gmsd:.4f}',
                                })
                            
                            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                                args_dic[args.loss.name] = f'{loss:.4f}'
                            t.set_postfix(**args_dic)
                    t.update()

        if self.writer is not None:
            self.writer.add_scalar('PSNR', np.mean(psnr_history), self.iteration)
            self.writer.add_scalar('SSIM', np.mean(ssim_history), self.iteration)
            self.writer.add_scalar('L1', np.mean(l1_history), self.iteration)
            self.writer.add_scalar('L2', np.mean(l2_history), self.iteration)
            if final:
                self.writer.add_scalar('GMSD', np.mean(gmsd_history), self.iteration)
                self.writer.add_scalar('LPIPS', np.mean(lpips_history), self.iteration)
            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                self.writer.add_scalar(args.loss.name, np.mean(loss_history), self.iteration)
        if args.report_to_mlflow:
            mlflow.log_metrics({'Valid-PSNR': float(np.mean(psnr_history)),
                                'Valid-PSNR-STD': float(np.std(psnr_history)),
                                'Valid-SSIM': float(np.mean(ssim_history)),
                                'Valid-SSIM-STD': float(np.std(ssim_history)),
                                'Valid-L1': float(np.mean(l1_history)),
                                'Valid-L2': float(np.mean(l2_history)),
                                }, step=self.iteration)
            if final:
                mlflow.log_metrics({'Valid-GMSD': float(np.mean(gmsd_history)),
                                    'Valid-GMSD-STD': float(np.std(gmsd_history)),
                                    'Valid-LPIPS': float(np.mean(lpips_history)),
                                    'Valid-LPIPS_STD': float(np.std(lpips_history)),
                                    }, step=self.iteration)
            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                mlflow.log_metrics({f'Valid-{args.loss.value.upper()}': float(np.mean(loss_history)),
                                    f'Valid-{args.loss.value.upper()}-STD': float(np.std(loss_history))
                                    }, step=self.iteration)
        
        if final:
            return np.mean(loss_history), np.std(loss_history), np.mean(psnr_history), np.std(psnr_history), \
                np.mean(ssim_history), np.std(ssim_history), np.mean(lpips_history),  np.std(lpips_history), \
                np.mean(gmsd_history), np.std(gmsd_history)
        else:
            return np.mean(loss_history), np.mean(psnr_history), np.mean(ssim_history)
                    
                
    
    def learning_rate_policy(self,):
        
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not self.iteration % args.learning_rate_policy_check_every
                and self.iteration - self.learning_rate_change_iter_nums[-1] > args.min_iters):
            [slope, _], [[var, _], _] = np.polyfit(self.loss_steps[-(args.learning_rate_slope_range //
                                                                    args.run_test_every):],
                                                   self.loss_values[-(args.learning_rate_slope_range //
                                                                  args.run_test_every):],
                                                   1, cov=True)

            # We take the the standard deviation as a measure
            std = np.sqrt(var)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -args.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10
                print(f'Iteration{self.learning_rate} learning rate updated: {self.learning_rate}')

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.iteration)
    

