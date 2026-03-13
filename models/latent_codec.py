import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import math
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .modules import DepthConvBlock, ResidualBlockUpsample2, ResidualBlockWithStride2

# v3_new
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def ste_round(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')
    
def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.branch1 = self.down  # legacy name for loading old checkpoints

    def forward(self, x):
        return self.down(x)
    
class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
        )
        self.branch1 = self.up  # legacy name for loading old checkpoints

    def forward(self, x):
        return self.up(x)


class AnalysisTransform(nn.Module):
    def __init__(self, ch_emd=32, channel=320):
        super().__init__()
        self.pre1 = nn.Sequential(
            nn.Conv2d(ch_emd, 128, kernel_size=3, padding=1),
        )
            
        self.pre2 = nn.Sequential(
            Downsample(320, 64)
        )
        self.analysis_transform = nn.Sequential(
            DepthConvBlock(192, 192),
            DepthConvBlock(192, 192),
            Downsample(192, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, channel),
        )

    def forward(self, latent, latent2):
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)
        x = self.analysis_transform(x)
        return x
    
class SynthesisTransform(nn.Module):
    def __init__(self, channel=320, channel_out=32) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),
            nn.Conv2d(320, channel_out, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.synthesis_transform(x)
        return x

class SynthesisTransform2(nn.Module):
    def __init__(self, channel=320, channel_out=32) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 192),
            nn.Conv2d(192, channel_out, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
        
    
class AuxDecoder(nn.Module):
    def __init__(self, ch_emd=32, channel=320) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),
            nn.Conv2d(320, ch_emd, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class HyperAnalysis(nn.Module):
    def __init__(self, channel=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            DepthConvBlock(channel, channel),
            DepthConvBlock(channel, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    
class HyperSynthesis(nn.Module):
    def __init__(self, channel=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            ResidualBlockUpsample2(channel // 2, channel // 2),
            ResidualBlockUpsample2(channel // 2, channel // 2),
            DepthConvBlock(channel//2, channel),
            DepthConvBlock(channel, channel),
        )

    def forward(self, x):
        x = self.increase(x)
        return x

class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out  

class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.branch1(x)
    
class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
        )

    def forward(self, x):
        context = self.block(x)
        return context

class LRP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Adapter(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)
    
class LatentCodec(nn.Module):
    def __init__(self, 
                 ch_emd=32,
                 channel=320, 
                 channel_out=320,
                 num_slices=5, 
                 max_support_slices=5, 
                 kernel_size=8,
                 drop_path_rate=0.1,
                 depths=[2,2,2,2,2],
                 num_heads=[12,12,12,12,12],
                 **kwargs
                 ):
        super().__init__()
        
        M = channel
        self.M = channel
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        num_per_slice = channel // self.num_slices
        
        self.g_a = AnalysisTransform(ch_emd, channel)
        self.g_s = SynthesisTransform(channel, channel_out)
        self.scale = SynthesisTransform2(channel, channel_out)
        self.prompt = SynthesisTransform(channel, 320)
        self.h_a = HyperAnalysis(channel)
        self.h_s = HyperSynthesis(channel)

        self.aux = AuxDecoder(ch_emd, channel)

        context_dim = M * 2
        self.adapter_in = nn.ModuleList(Adapter(in_ch=M, out_ch=context_dim) for i in range(4))
        self.g_c = SpatialContext(in_ch=context_dim)
        self.adapter_out = nn.ModuleList(Adapter(in_ch=context_dim, out_ch=M * 2) for i in range(4))
        self.LRP = nn.ModuleList(LRP(in_ch=M * 2, out_ch=M) for i in range(4))

        self.entropy_bottleneck = EntropyBottleneck(channel // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        self.apply(self._init_weights)  
    
    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)
            m2 = m2[:height, :width]
            m2 = torch.unsqueeze(m2, 0)
            m2 = torch.unsqueeze(m2, 0)

            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)
            m3 = m3[:height, :width]
            m3 = torch.unsqueeze(m3, 0)
            m3 = torch.unsqueeze(m3, 0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=False):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)
    
    def sequeeze_with_mask(self, latent, mask):
        latent_group_1, latent_group_2, latent_group_3, latent_group_4 = latent.chunk(4, 1)
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent_sequeeze = latent_group_1 * mask_group_1 + latent_group_2 * mask_group_2 + latent_group_3 * mask_group_3 + latent_group_4 * mask_group_4
        return latent_sequeeze
    
    def unsequeeze_with_mask(self, latent_sequeeze, mask):
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent = torch.cat((latent_sequeeze * mask_group_1, latent_sequeeze * mask_group_2, latent_sequeeze * mask_group_3, latent_sequeeze * mask_group_4), dim=1)
        return latent
    
    def compress_group_with_mask(self, gaussian_conditional, latent, scales, means, mask, symbols_list, indexes_list):
        latent_squeeze = self.sequeeze_with_mask(latent, mask)
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = gaussian_conditional.quantize(latent_squeeze, "symbols", means_squeeze)
        symbols_list.extend(latent_squeeze_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat
    
    def decompress_group_with_mask(self, gaussian_conditional, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        latent_squeeze_hat = torch.Tensor(latent_squeeze_hat).reshape(scales_squeeze.shape).to(scales.device)
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat
    
    def forward_with_mask(self, latent, scales, means, mask):
        latent_squeeze = self.sequeeze_with_mask(latent, mask)
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        
        _, y_likelihoods = self.gaussian_conditional(latent_squeeze, scales_squeeze, means=means_squeeze)
        latent_hat = ste_round(latent_squeeze - means_squeeze) + means_squeeze
        y_likelihoods = self.unsequeeze_with_mask(y_likelihoods, mask)
        
        latent_hat = self.unsequeeze_with_mask(latent_hat, mask)
        return latent_hat, y_likelihoods

    def scale_with_mask(self, scales_1, scales_2, scales_3, scales_4, mask_0, mask_1, mask_2, mask_3):
        scales_1 = self.sequeeze_with_mask(scales_1, mask_0)
        scales_1 = self.unsequeeze_with_mask(scales_1, mask_0)

        scales_2 = self.sequeeze_with_mask(scales_2, mask_1)
        scales_2 = self.unsequeeze_with_mask(scales_2, mask_1)

        scales_3 = self.sequeeze_with_mask(scales_3, mask_2)
        scales_3 = self.unsequeeze_with_mask(scales_3, mask_2)

        scales_4 = self.sequeeze_with_mask(scales_4, mask_3)
        scales_4 = self.unsequeeze_with_mask(scales_4, mask_3)

        scales = scales_1 + scales_2 + scales_3 + scales_4
        return scales

    def forward(self, latent, latent2):
        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        # torch.backends.cudnn.deterministic = True
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        
        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)
        
        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0, y_likelihoods_0 = self.forward_with_mask(y, scales_0_supp, means_0_supp, mask_0)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp
        
        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1, y_likelihoods_1 = self.forward_with_mask(y, scales_1_supp, means_1_supp, mask_1)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2, y_likelihoods_2 = self.forward_with_mask(y, scales_2_supp, means_2_supp, mask_2)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp
        
        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3, y_likelihoods_3 = self.forward_with_mask(y, scales_3_supp, means_3_supp, mask_3)
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp
        
        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        y_likelihoods = y_likelihoods_0 + y_likelihoods_1 + y_likelihoods_2 + y_likelihoods_3 
        # torch.backends.cudnn.deterministic = False
        
        latent_scales = self.scale_with_mask(scales_0_supp, scales_1_supp, scales_2_supp, scales_3_supp, mask_0, mask_1, mask_2, mask_3)
        mean = self.g_s(y_hat)
        scale = self.scale(latent_scales)
        res = self.aux(y_hat)
        prompt = self.prompt(y_hat)
        
        return {
            "mean": mean,
            "scale": scale,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "res": res,
            "prompt": prompt
        }

    def compress(self, latent, latent2):
        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_0_supp, means_0_supp, mask_0, symbols_list, indexes_list)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_1_supp, means_1_supp, mask_1, symbols_list, indexes_list)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_2_supp, means_2_supp, mask_2, symbols_list, indexes_list)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_3_supp, means_3_supp, mask_3, symbols_list, indexes_list)
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):

        torch.backends.cudnn.deterministic = True
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        B, C, H, W = z_hat.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C * 2, H * 4, W * 4, device=z_hat.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.decompress_group_with_mask(self.gaussian_conditional, scales_0_supp, means_0_supp, mask_0, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.decompress_group_with_mask(self.gaussian_conditional, scales_1_supp, means_1_supp, mask_1, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.decompress_group_with_mask(self.gaussian_conditional, scales_2_supp, means_2_supp, mask_2, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.decompress_group_with_mask(self.gaussian_conditional, scales_3_supp, means_3_supp, mask_3, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3

        torch.backends.cudnn.deterministic = False

        latent_scales = self.scale_with_mask(scales_0_supp, scales_1_supp, scales_2_supp, scales_3_supp, mask_0, mask_1, mask_2, mask_3)
        mean = self.g_s(y_hat)
        scale = self.scale(latent_scales)
        res = self.aux(y_hat)
        prompt = self.prompt(y_hat)

        return mean, scale, res, prompt

if __name__ == '__main__':
    model = LatentCodec()
    x = torch.randn(1, 4, 32, 32)
    y = torch.randn(1, 320, 16, 16)
    model(x,y)
