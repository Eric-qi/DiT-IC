import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union
from diffusers import DPMSolverMultistepScheduler



def make_1step_sched(pretrained_path, time, device='cuda'):
    noise_scheduler_1step = DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device=device)
    # noise_scheduler_1step.timesteps = torch.tensor([time], dtype=torch.long, device=device)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
    noise_scheduler_1step.betas = noise_scheduler_1step.betas.to(device)

    return noise_scheduler_1step

class MyDDPMScheduler:
    def __init__(self, base_scheduler: DPMSolverMultistepScheduler, mode='prev_sample', device='cuda'):
        self.base_scheduler = base_scheduler
        self.mode = mode
        print(f'========mode is #{mode}#======')
        if 'adapt' not in self.mode:
            self.adaptor = AdaptiveNoiseScheduler(mode).to(device)
        else:
            self.adaptor = None
        
    def step(self, model_output, scale, timestep, sample, **kwargs):
        if self.mode == "epsilon":
            # DPM-Solver and DPM-Solver++ only need the "mean" output.
            sigma = self.base_scheduler.sigmas[0]
            alpha_t, sigma_t = self.base_scheduler._sigma_to_alpha_sigma_t(sigma)
            x0_pred = (sample - sigma_t * model_output) / alpha_t
            return x0_pred
        elif self.mode == "sample":
            x0_pred = model_output
            return x0_pred
        elif self.mode == "flow_prediction":
            sigma_t = self.base_scheduler.sigmas[0]
            x0_pred = sample - sigma_t * model_output
            return x0_pred
        elif self.mode == "v_prediction":
            sigma = self.sigmas[0]
            alpha_t, sigma_t = self.base_scheduler._sigma_to_alpha_sigma_t(sigma)
            x0_pred = alpha_t * sample - sigma_t * model_output
            return x0_pred
        else:
            shift, scale = self.adaptor(sample, scale)
            sigma_t = self.base_scheduler.sigmas[0]
            adaptive_sigma = modulate(sigma_t, shift, scale)
            x0_pred = sample - adaptive_sigma * model_output
            return x0_pred
    
    


class AdaptiveNoiseScheduler(nn.Module):
    def __init__(self, mode='scale', **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, x, scale=None):
        """
        Input: x: (N, C, H, W)
        Return: scale: (N, C, H, W)
        """
        if self.mode=='value':
            a = 2.0
            linear_part = (0.6 / a) * x.abs() + 0.4
            flat_part = torch.ones_like(x)
            scale = torch.where(x.abs() < a, linear_part, flat_part) - 1
            shift = 0.
        elif self.mode=='scale':
            a = 4.0
            linear_part = (0.6 / a) * scale.abs() + 0.4
            flat_part = torch.ones_like(scale)
            scale = torch.where(scale.abs() < a, linear_part, flat_part) - 1
            shift = 0.
        else:
            raise ValueError
        return shift, scale

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents