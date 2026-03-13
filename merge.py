import torch

import os
import yaml
import importlib
from pathlib import Path

from eval.compress_utils import *
from eval.testing_utils import parse_args_testing




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
    
    config = load_config(args.config_path)
    config["model"]["params"]["codec_path"] = args.codec_path
    config["model"]["params"]["use_ema"] = args.use_ema
    config["model"]["params"]["use_merge"] = False
    config["model"]["params"]["training"] = False
    
    params = config.get("model", {}).get("params", {})
    net = codec_models(config['model']['model_type'], 'cuda', **params)
    net.cuda().eval()
    net.merge_lora()
    
    total_params = sum(
        p.numel() for name, p in net.named_parameters()
        if not name.startswith("loss")
    )
    print("Total parameters: {:.3f}B".format(total_params / 1e9))
    net.codec.update(force=True)
    
    if args.use_ema:
        merge_path = Path(args.codec_path).with_name("merge_ema.pt")
    else:
        merge_path = Path(args.codec_path).with_name("merge.pt")
    torch.save(net.state_dict(), merge_path)
    

if __name__ == "__main__":
    args = parse_args_testing()
    main(args)
