import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.optim as optim
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


import os
import sys
import yaml
import json
import random
import logging
import argparse
import importlib
import numpy as np
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict

from datasets.image import ImageFolder




def codec_models(model, device, **kwargs):
    model_map = {
        'Codec': 'models.DiT_IC.Codec',
    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(device, model, **kwargs)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict) or isinstance(obj, torch.nn.modules.container.OrderedDict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def do_train(rank, local_rank, world_size, train_config):
    device = f'cuda:{rank}'

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"{train_config['train']['output_dir']}/tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)

    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # Create model:
    params = train_config.get("model", {}).get("params", {})
    model = codec_models(train_config['model']['model_type'], device, **params).to(device)
    
    if rank==0:
        ema = deepcopy(model.get_state_dicts_for_save())

    if rank == 0:
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.get_trainable_params()) / 1e6:.2f}M")
        logger.info(f"Trainable Codec Parameters: {sum(p.numel() for p in model.codec.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
    opt = torch.optim.AdamW(model.get_trainable_params(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    opt_disc = torch.optim.AdamW(model.loss.discriminator.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=train_config['optimizer']['beta_vae'])
    opt_aux = optim.AdamW(model.get_trainable_auxloss(), lr=train_config['optimizer']['aux_lr'], weight_decay=0)
    
    milestones = train_config['optimizer']['step_lr']
    gammas = [0.5, 0.2, 0.1]

    def lr_lambda(epoch):
        factor = 1.0
        for m, g in zip(milestones, gammas):
            if epoch >= m:
                factor *= g
        return factor

    # scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
    scheduler_disc = LambdaLR(opt_disc, lr_lambda=lr_lambda)
    
    # scheduler = MultiStepLR(opt, milestones=train_config['optimizer']['step_lr'], gamma=0.5)
    # scheduler_disc = MultiStepLR(opt_disc, milestones=train_config['optimizer']['step_lr'], gamma=0.5)
    
    model = DDP(model, device_ids=[rank], static_graph=True)
    # Setup data
    train_transforms = transforms.Compose([
            ResizeIfSmall(train_config['data']['path_size']),
            transforms.RandomCrop(train_config['data']['path_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    dataset = ImageFolder(train_config['data']['data_path'], train_config['data']['im_exts'], transform=train_transforms)
    
    # batch_size_per_gpu = int(train_config['train']['global_batch_size'] / torch.cuda.device_count())
    batch_size_per_gpu = train_config['train']['global_batch_size'] // world_size
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=my_worker_init_fn,
        sampler=sampler,
    )

    if rank == 0: 
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {train_config['train']['global_batch_size']} global batch size")
    
    if 'valid_path' in train_config['data']:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.CenterCrop(train_config['data']['path_size']),
        ])
        
        valid_dataset = ImageFolder(train_config['data']['valid_path'], train_config['data']['im_exts'], transform=val_transforms)
    
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        if rank == 0: 
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")
    
    # Prepare models for training:
    if rank==0:
        update_ema(ema, model.module.get_state_dicts_for_save(), decay=0.999)  # Ensure EMA is initialized with  weights
    model.train()

    train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=os.path.getmtime)
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            
            model.module.load_ckpt_dict(latest_checkpoint, use_ema=False)
            model.to(device)
            model.train()
            
            opt.load_state_dict(checkpoint['opt'])
            opt_disc.load_state_dict(checkpoint['opt_disc'])
            opt_aux.load_state_dict(checkpoint['opt_aux'])

            ema = checkpoint['ema']
            ema = move_to_device(checkpoint['ema'], device)
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                scheduler_disc.load_state_dict(checkpoint["scheduler_disc"])
            else:
                scheduler = MultiStepLR(opt, milestones=train_config['optimizer']['step_lr'], gamma=0.5, last_epoch=train_steps)
                scheduler_disc = MultiStepLR(opt_disc, milestones=train_config['optimizer']['step_lr'], gamma=0.5, last_epoch=train_steps)
    
            if rank == 0: 
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            train_steps = 0
            if rank == 0: 
                logger.info("No checkpoint found. Starting training from scratch.")

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    running_aux_loss = 0
    running_sdis_loss = 0
    running_dis_loss = 0

    best_loss = float("inf")
    start_time = time()
    
    while True:
        # sampler.set_epoch(train_steps)
        epoch = train_steps // len(loader)  # ★DDP: ensure shuffle
        sampler.set_epoch(epoch)


        model.module.loss.kl_weight = train_config['train']['lrd']
        for x in loader:
            x = x.to(device)

            # if optimizer_idx == 0: train codec
            output_image, rate, z, aux_feature, kl_loss = model(x)
            # if train_steps < train_config['model']['params']['loss_params']['disc_start']:
            #     loss, log_dict = model.module.loss(x, output_image, rate, -1, train_steps,
            #                                 last_layer=model.module.get_last_layer(), split="train", z=z, aux_feature=aux_feature, 
            #                                 enc_last_layer=model.module.get_vf_last_layer())
            # else:
            #     loss, log_dict = model.module.loss(x, output_image, rate, 0, train_steps,
            #                                 last_layer=model.module.get_last_layer(), split="train", z=z, aux_feature=aux_feature, 
            #                                 enc_last_layer=model.module.get_vf_last_layer())

            loss, log_dict = model.module.loss(x, output_image, rate, 0, train_steps,
                                            last_layer=model.module.get_last_layer(), split="train", z=z, aux_feature=aux_feature, 
                                            enc_last_layer=model.module.get_vf_last_layer())
            
            loss += kl_loss
            opt.zero_grad()
            loss.backward()
            if 'max_grad_norm' in train_config['optimizer']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            scheduler.step()
            # aux loss
            aux_loss = model.module.codec.aux_loss()
            opt_aux.zero_grad()
            aux_loss.backward()
            opt_aux.step()
            if rank == 0: 
                update_ema(ema, model.module.get_state_dicts_for_save(), decay=0.999)  # Ensure EMA is initialized with  weights

            # if optimizer_idx == 1: train the discriminator
            opt_disc.zero_grad()
            # if train_steps >= train_config['model']['params']['loss_params']['disc_start']:
            #     discloss, log_dict_disc = model.module.loss(x, output_image, rate, 1, train_steps,
            #                                     last_layer=model.module.get_last_layer(), split="train", enc_last_layer=model.module.get_vf_last_layer())
            #     discloss.backward()
                
            #     if 'max_grad_norm' in train_config['optimizer']:
            #         torch.nn.utils.clip_grad_norm_(model.module.loss.discriminator.parameters(), train_config['optimizer']['max_grad_norm'])
            #     opt_disc.step()

            # else:
            #     discloss = torch.tensor(0.0, device=x.device)

            discloss, log_dict_disc = model.module.loss(x, output_image, rate, 1, train_steps,
                                            last_layer=model.module.get_last_layer(), split="train", enc_last_layer=model.module.get_vf_last_layer())
            discloss.backward()
            
            if 'max_grad_norm' in train_config['optimizer']:
                torch.nn.utils.clip_grad_norm_(model.module.loss.discriminator.parameters(), train_config['optimizer']['max_grad_norm'])
            opt_disc.step()

            scheduler_disc.step()

            # Log loss values:
            running_loss += loss.item()
            running_aux_loss += aux_loss.item()
            running_sdis_loss += kl_loss.item()
            running_dis_loss += discloss.item()
            log_steps += 1
            train_steps += 1

            
            
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # d loss
                avg_dis_loss = torch.tensor(running_dis_loss / log_steps, device=device)
                dist.all_reduce(avg_dis_loss, op=dist.ReduceOp.SUM)
                avg_dis_loss = avg_dis_loss.item() / dist.get_world_size()
                # aux loss
                avg_aux_loss = torch.tensor(running_aux_loss / log_steps, device=device)
                dist.all_reduce(avg_aux_loss, op=dist.ReduceOp.SUM)
                avg_aux_loss = avg_aux_loss.item() / dist.get_world_size()
                # self-dist loss
                avg_sdis_loss = torch.tensor(running_sdis_loss / log_steps, device=device)
                dist.all_reduce(avg_sdis_loss, op=dist.ReduceOp.SUM)
                avg_sdis_loss = avg_sdis_loss.item() / dist.get_world_size()

                lr = opt.param_groups[0]['lr']
                if rank == 0: 
                    logger.info(f"(step={train_steps:07d}, lr = {lr:.6e}) Train Loss: {avg_loss:.4f},  Dis: {avg_dis_loss:.4f}, Aux: {avg_aux_loss:.4f}, SDIS: {avg_sdis_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    log = {k: float(v) for k, v in log_dict.items()}
                    logger.info(json.dumps(log, indent=2))
                    if train_steps > train_config['model']['params']['loss_params']['disc_start']:
                        dis_log = {k: float(v) for k, v in log_dict_disc.items()}
                        logger.info(json.dumps(dis_log, indent=2))
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_aux_loss = 0
                running_sdis_loss = 0
                running_dis_loss = 0
                log_steps = 0
                start_time = time()

                # Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if rank == 0: 
                        logger.info(f"Start evaluating at step {train_steps}")

                    test_loss = evaluate(model, valid_loader, device, logger if rank==0 else None, writer if rank==0 else None)

                    if rank == 0:
                        is_best = test_loss < best_loss
                        best_loss = min(test_loss, best_loss)
                        if is_best and train_steps > 0.7 * train_config['train']['max_steps']:
                            checkpoint = {
                                "model": model.module.get_state_dicts_for_save(),
                                "ema": ema,
                                "opt": opt.state_dict(),
                                "opt_disc": opt_disc.state_dict(),
                                "opt_aux": opt_aux.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scheduler_disc": scheduler_disc.state_dict(),
                                "config": train_config,
                            }
                            checkpoint_path = f"{checkpoint_dir}/best_ckpt.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved best checkpoint to {checkpoint_path} at step {train_steps}")
                    #dist.barrier(device_ids=[local_rank])
                    dist.barrier()
                    model.train()


            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if rank == 0: 
                    checkpoint = {
                        "model": model.module.get_state_dicts_for_save(),
                        "ema": ema,
                        "opt": opt.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "opt_aux": opt_aux.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scheduler_disc": scheduler_disc.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            
                model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if rank == 0: 
        logger.info("Done!")

@torch.no_grad()
def evaluate(model, test_loader, device, logger=None, writer=None):
    model.eval()
    loss = AverageMeter()
    
    mse_loss = AverageMeter()
    psnr_loss = AverageMeter()
    lpips_loss = AverageMeter()
    bpp_loss = AverageMeter()
    aux_loss = AverageMeter()
    all_loss = AverageMeter()
    
    for x in test_loader:
        x = x.to(device)

        output_image, rate, z, aux_feature, kl_loss = model(x)
        loss, rec_loss, psnr, rate_loss, p_loss = model.module.loss(x, output_image, rate, -1, -1, split="eval")
        
        all_loss.update(loss)
        mse_loss.update(rec_loss)
        psnr_loss.update(psnr)
        lpips_loss.update(p_loss)
        bpp_loss.update(rate_loss)
        aux_loss.update((model.module.codec if hasattr(model, 'module') else model.codec).aux_loss())

    if logger is not None:
        logger.info(
            f"Test Average losses: "
            f"Loss: {all_loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"PSNR: {psnr_loss.avg:.2f} | "
            f"LPIPS: {lpips_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f}\n"
        )

    return all_loss.avg
    
    
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.999):
#     """
#     Step the EMA model towards the current model.
#     """
#     for part_key in ema_model.keys():
#         ema_part = ema_model[part_key]
#         model_part = model.get(part_key, {})

#         for param_key in ema_part.keys():
#             if param_key in model_part:
#                 ema_tensor = ema_part[param_key]
#                 model_tensor = model_part[param_key]
#                 if torch.is_floating_point(ema_tensor):
#                     ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
#                 else:
#                     ema_tensor.copy_(model_tensor)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    for part_key in ema_model.keys():
        ema_part = ema_model[part_key]
        model_part = model.get(part_key, {})

        for param_key in ema_part.keys():
            if param_key in model_part:
                ema_tensor = ema_part[param_key]
                model_tensor = model_part[param_key]

                # 如果 shape 不一致，直接替换
                if ema_tensor.shape != model_tensor.shape:
                    ema_part[param_key] = model_tensor.clone()
                    continue

                # 正常 EMA 更新
                if torch.is_floating_point(ema_tensor):
                    ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
                else:
                    ema_tensor.copy_(model_tensor)
            else:
                # 如果模型里没有这个参数，保持不变
                continue

        # 如果模型新增了参数，这里同步补上
        for param_key, model_tensor in model_part.items():
            if param_key not in ema_part:
                ema_part[param_key] = model_tensor.clone()



def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

class ResizeIfSmall:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if w < self.min_size[0] or h < self.min_size[1]:
            new_w = max(w, self.min_size[0])
            new_h = max(h, self.min_size[1])
            return transforms.Resize((new_h, new_w))(img)
        return img
    
def create_logger(logging_dir):
    """
    Create a logger that writes color logs to stdout and plain logs to file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if len(logger.handlers) > 0:
        return logger

    if dist.get_rank() == 0:
        os.makedirs(logging_dir, exist_ok=True)

        # Terminal (stdout) handler with color
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File handler without color
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def set_seed(seed=903):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank

def cleanup_ddp():
    dist.destroy_process_group()

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    train_config = load_config(args.config)
    set_seed(train_config["train"]["global_seed"])

    rank, world_size, local_rank = setup_ddp()
    do_train(rank, local_rank, world_size, train_config)
    cleanup_ddp()
    