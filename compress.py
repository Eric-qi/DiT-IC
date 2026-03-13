import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import (
    FrechetInceptionDistance,
    KernelInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)

import os
import math
import glob
import yaml
import pyiqa
import importlib
import numpy as np
from PIL import Image
from accelerate.utils import set_seed

from eval.compress_utils import *
from eval.testing_utils import parse_args_testing
from eval._update_patch_fid import update_patch_fid




def codec_models(model, device, **kwargs):
    model_map = {
        'Codec': 'models.DiT_IC.Codec',

    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(device, model, **kwargs)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    if not os.path.exists(bin_path): os.makedirs(bin_path)
    output = os.path.join(bin_path, img_name)
    with Path(output).open("wb") as f:
        write_body(f, shape, output_dict["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = read_body(f)
    with torch.no_grad():
        out_img = net.decompress(strings, shape)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().detach()
    return out_img


def main(args):

    if args.seed is not None:
        set_seed(args.seed)
    
    device = torch.device("cuda")
    
    # define Non-Reference Model
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device) # [0-1] input
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device) # [0-1] input
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device) # [0-1] input
    # metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device) # [0-1] input
    
    # define Reference Model
    metric_paired_dict = {}
    metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device) # [0-1] input
    metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device) # [0-1] input
    metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device) # [0-1] input
    metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device) # [0-1] input
    
    # define Distribution Model
    fid_metric = FrechetInceptionDistance().to(device)
    kid_metric = KernelInceptionDistance().to(device)
    
    # load model
    config = load_config(args.config_path)
    config["model"]["params"]["codec_path"] = args.codec_path
    config["model"]["params"]["use_ema"] = args.use_ema
    config["model"]["params"]["use_merge"] = args.use_merge
    config["model"]["params"]["training"] = False
    
    params = config.get("model", {}).get("params", {})
    net = codec_models(config['model']['model_type'], 'cuda', **params)
    net.cuda().eval()
    net.codec.update(force=True)

    transform = transforms.Compose([
        # transforms.Resize((512, 512)),    
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    bpp = []
    images = glob.glob(args.img_path + '/*.png')
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')
    
    result = {}
    for img_path in images:
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect') # [-1,1]

        with torch.no_grad():
            if args.entropy_estimation:
                N, _, H, W = img.size()
                num_pixels = N * H * W
                out_img, rate, _, _, _ = net(img_padded)
                
                rate = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in rate.values()
                )
                out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
                out_img = (out_img * 0.5 + 0.5).float().detach()
            else:
                rate = compress_one_image(net, args.bin_path, ori_h, ori_w, fname, img_padded)
                out_img = decompress_one_image(net, args.bin_path, ori_h, ori_w, fname)
        
        out_img = out_img.clamp(0.0, 1.0) # [0, 1]
        gt_img = (img * 0.5 + 0.5).float().detach()
        
        # compute metrics
        print(f'=============={fname}==============')
        for key, metric in metric_dict.items():
            value = metric(out_img).item()
            print(key, value)
            result[key] = result.get(key, 0) + value
        
        update_patch_fid(gt_img, out_img, fid_metric=fid_metric, kid_metric=kid_metric)  
        
        for key, metric in metric_paired_dict.items():
            value = metric(out_img, gt_img).item()
            print(key, value)
            result[key] = result.get(key, 0) + value
        
        try:
            bpp.append(rate.cpu().numpy())
        except:
            bpp.append(rate)
        print('[BPP]', rate)

        if args.save_img:
            output_pil = transforms.ToPILImage()(out_img[0].cpu()) 
            output_pil.save(outf)
    
    print(f'==============The Avg. Results w/ {str(len(images))} images==============')
    result['fid'] = float(fid_metric.compute())
    
    if len(images) > 50:
        kid_tuple = kid_metric.compute()
        result['kid_mean'], result['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])
    
    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        elif key == 'kid_mean' or key == 'kid_std':
            print(f"{key}: {res:.7f}")
            print_results.append(f"{key}: {res:.7f}")
        else:
            print(f"{key}: {res/len(images):.5f}")
            print_results.append(f"{key}: {res/len(images):.5f}")
            
    print('\n[Average BPP]', np.mean(bpp))
    

if __name__ == "__main__":
    args = parse_args_testing()
    if args.save_img:
        if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
        if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    main(args)
