import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
import time
import importlib
from peft import get_peft_model, LoraConfig
from diffusers import AutoencoderDC, SanaTransformer2DModel, DPMSolverMultistepScheduler


from .losses import LPIPSWithDiscriminator
from .foundation_models import aux_foundation_model
from .scheduler import make_1step_sched, MyDDPMScheduler, randn_tensor

sys.path.append("..")
from ELIC.elic_official import ELIC


# we lora for decoder instead of encoder 
def filter_supported_modules(model):
    import re, torch.nn as nn
    pattern = re.compile(r"^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv_inverted|conv_point|to_k|to_q|to_v|to_out\.0)$")
    supported = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return [n for n, m in model.named_modules() if pattern.match(n) and isinstance(m, supported)]


def latent_models(model, **kwargs):
    model_map = {
        'Codec': 'models.latent_codec.LatentCodec',
    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)


class LatentConditionAlignment(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 2304,
        num_tokens: int = 77,
        mode: str = "mlp",  # ['mlp', 'transformer']
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        use_clip_contrast: bool = False,
        clip_embed_dim: int = 768
    ):
        super().__init__()
        assert mode in ["mlp", "transformer"]
        self.mode = mode
        self.num_tokens = num_tokens
        self.fixed_tokens = 48
        self.embed_dim = embed_dim
        self.use_clip_contrast = use_clip_contrast
        self.clip_embed_dim = clip_embed_dim

        self.pattern = nn.Parameter(torch.randn(1, self.num_tokens - self.fixed_tokens, in_channels))

        # --- latent-condition alignment module---
        if mode == "mlp":
            self.align = nn.Sequential(
                nn.Linear(in_channels, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.proj_in = nn.Linear(in_channels, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=4 * embed_dim,
                batch_first=True,
                activation="gelu"
            )
            self.align = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        # ---pretrained-embedding alignment module ---
        if use_clip_contrast:
            self.proj_clip = nn.Linear(embed_dim, clip_embed_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, latent, text_embed=None, image_embed=None):
        """
        latent: [B, C, H, W]
        text_embed: [B, D]  (CLIP text embedding, Pre-stored, optional)
        image_embed: [B, D] (CLIP image embedding, Pre-stored, optional)
        return:
            pos_caption_enc: [B, num_tokens, embed_dim]
            clip_align_loss (optional): scalar
        """
        B, C, H, W = latent.shape

        # [B, C, H*W] → [B, C, 48]
        x = latent.flatten(2)
        x = F.adaptive_avg_pool1d(x, self.fixed_tokens)
        x = x.transpose(1, 2)  # [B, 48, C]

        # repeat pattern 以匹配 batch
        pattern = self.pattern.expand(B, -1, -1)  # [B, 77-48, C]

        # 拼接得到完整 token 序列
        x = torch.cat([x, pattern], dim=1)  # [B, 77, C]

        # Step 3: 空间-语义映射
        if self.mode == "mlp":
            pos_caption_enc = self.align(x)
        else:
            x = self.proj_in(x)
            pos_caption_enc = self.align(x)

        # Step 4: CLIP-style 对齐损失
        clip_align_loss = None
        if self.use_clip_contrast and (text_embed is not None or image_embed is not None):
            latent_global = pos_caption_enc.mean(dim=1)  # [B, embed_dim]
            latent_global = F.normalize(self.proj_clip(latent_global), dim=-1)

            clip_target = image_embed if text_embed is not None else image_embed
            clip_target = F.normalize(clip_target, dim=-1)
            logit_scale = self.logit_scale.exp()

            # 双向对比学习
            logits_per_latent = logit_scale * latent_global @ clip_target.t()
            logits_per_target = logits_per_latent.t()
            labels = torch.arange(B, device=latent.device)

            clip_align_loss = (
                F.cross_entropy(logits_per_latent, labels) +
                F.cross_entropy(logits_per_target, labels)
            ) / 2

        return (pos_caption_enc, clip_align_loss) if self.use_clip_contrast else pos_caption_enc



class Codec(torch.nn.Module):
    def __init__(self, 
                 device='cuda',
                 model_type='Codec',
                 time=999,
                 codec_mode='self_dist',
                 DiT_mode='scale',
                 dit_path=None, 
                 elic_path=None,
                 codec_path=None,
                 use_ema=False,
                 use_merge=False,
                 training=False,
                 DiT_params=None,
                 vae_params=None,
                 codec_param=None,
                 loss_params=None,
                 **kwargs
                 ):
        super().__init__()
        
        self.DiT_mode = DiT_mode
        self.codec_mode = codec_mode
        self.model_type = model_type
        self.time = time
        
        # DiT
        if use_merge:
            vae_config = AutoencoderDC.load_config(dit_path, subfolder="vae")
            self.vae = AutoencoderDC.from_config(vae_config)
        else:
            self.vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")
        for param in self.vae.parameters():
            param.requires_grad = False
            
        if DiT_params["DiT"]:
            self.build_DiT(dit_path, DiT_params["channel_in"], device, use_merge)
        
        # Latent Codec
        self.codec = latent_models(model_type, **codec_param)

        # Latent Prompt Condition
        self.prompter = LatentConditionAlignment(
            in_channels = 320,
            embed_dim = 2304, # change according to used embedding model.
            num_tokens = 77,
            mode = "mlp",  # ['mlp', 'transformer']
            transformer_depth = 2,
            transformer_heads = 8,
            use_clip_contrast = False,
            clip_embed_dim = 768 # 1024, xxx
        )
        
        # Aux Encoder
        model = ELIC()
        if not use_merge:
            checkpoint = torch.load(elic_path)
            model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        
        # Loss
        if loss_params["use_vf"] is not None:
            self.use_vf = loss_params["use_vf"]
            self.vf_latent = loss_params["latent"]
            self.foundation_model = aux_foundation_model(loss_params["use_vf"], device)
            if loss_params["latent"]:
                vf_feature_dim = self.foundation_model.feature_dim
                self.linear_proj = torch.nn.Conv2d(vf_feature_dim, codec_param["channel"], kernel_size=1, bias=True)
                if loss_params["reverse_proj"]:
                    self.linear_proj = torch.nn.Conv2d(codec_param["channel"], vf_feature_dim, kernel_size=1, bias=False)
        else:
            self.use_vf = None
        self.loss = LPIPSWithDiscriminator(device, **loss_params)
        
        # load chekcpoint
        if use_merge:
            assert codec_path is not None
            state_dict = torch.load(codec_path, map_location="cpu")
            codec_state = {}
            model_state = {}

            for k, v in state_dict.items():
                if k.startswith("codec."):
                    codec_state[k[6:]] = v
                else:
                    model_state[k] = v
            self.codec.load_state_dict(codec_state)
            self.load_state_dict(model_state, strict=False)
            
            if training and vae_params["vae_lora"]:
                self.build_vae_lora(vae_params["lora_rank_vae"], vae_params["lora_alpha_vae"])
            if training and DiT_params["DiT_lora"]:
                self.build_DiT_lora(DiT_params["lora_rank_DiT"], DiT_params["lora_alpha_DiT"], DiT_params["channel_in"])
        else:
            if vae_params["vae_lora"]:
                self.build_vae_lora(vae_params["lora_rank_vae"], vae_params["lora_alpha_vae"])
            if DiT_params["DiT_lora"]:
                self.build_DiT_lora(DiT_params["lora_rank_DiT"], DiT_params["lora_alpha_DiT"], DiT_params["channel_in"])

            if codec_path is not None:
                self.load_ckpt_dict(codec_path, use_ema)
            
        print("Init Done")
    
    def merge_lora(self):
        self.vae = self.vae.merge_and_unload()
        self.DiT = self.DiT.merge_and_unload()
        
    def load_ckpt_dict(self, codec_path, use_ema=False):
        print("[LoRA & Latent Codec & Decoder]: Loading Pretrained Weights ......")
        if use_ema:
            dit = torch.load(codec_path, map_location="cpu")["ema"]
        else:
            dit = torch.load(codec_path, map_location="cpu")["model"]

        _dit_codec = self.codec.state_dict()
        for k in dit["state_dict_codec"]:
            _dit_codec[k] = dit["state_dict_codec"][k]
        self.codec.load_state_dict(_dit_codec)
        self.codec.update()

        _dit_vae = self.vae.state_dict()
        for k in dit["state_dict_vae"]:
            _dit_vae[k] = dit["state_dict_vae"][k]
        self.vae.load_state_dict(_dit_vae)

        _dit_DiT = self.DiT.state_dict()
        for k in dit["state_dict_DiT"]:
            _dit_DiT[k] = dit["state_dict_DiT"][k]
        self.DiT.load_state_dict(_dit_DiT)
        
        self.prompter.load_state_dict(dit["state_dict_prompter"])

        if self.use_vf is not None and self.vf_latent:
            self.linear_proj.load_state_dict(dit["state_dict_vf"])
            
        
    def build_DiT(self, dit_path, channel, device='cuda', use_merge=False):
        scheduler = make_1step_sched(dit_path, self.time, device)
        self.sched = MyDDPMScheduler(scheduler, self.DiT_mode, device)

        if use_merge:
            dit_config = SanaTransformer2DModel.load_config(dit_path, subfolder="transformer")
            self.DiT = SanaTransformer2DModel.from_config(dit_config)
        else:
            self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer")
            # self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer_512")

        self.register_buffer("timesteps", torch.tensor([self.time], dtype=torch.long))
        
        for name, param in self.DiT.named_parameters():
            param.requires_grad = False
        
    def build_vae_lora(self, lora_rank_vae, lora_alpha_vae):
        vae_lora_config = LoraConfig(
            r=lora_rank_vae,
            lora_alpha=lora_alpha_vae,
            target_modules=filter_supported_modules(self.vae),
            bias="none",
            init_lora_weights="gaussian",
        )
        self.vae = get_peft_model(self.vae, vae_lora_config)
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        print("VAE-LoRA Done")

    def build_DiT_lora(self, lora_rank_DiT, lora_alpha_DiT, channel):
        target_modules_DiT = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj", "conv_inverted", "conv_point"
        ]
        DiT_lora_config = LoraConfig(
            r=lora_rank_DiT,
            lora_alpha=lora_alpha_DiT,
            target_modules=target_modules_DiT,
            bias="none",
            init_lora_weights="gaussian",
        )
        self.DiT = get_peft_model(self.DiT, DiT_lora_config)
        
        for param in self.DiT.parameters():
            param.requires_grad = False
        
        for name, param in self.DiT.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                
        print("DiT-LoRA Done")
        
    def get_trainable_auxloss(self):
        aux_params = [p for n, p in self.codec.named_parameters() if n.endswith(".quantiles") and p.requires_grad]
        return aux_params
        
    def get_trainable_params(self):
        params_to_optimize = []
        
        # Codec all params
        codec_params = [p for n, p in self.codec.named_parameters() if not n.endswith(".quantiles") and p.requires_grad]
        params_to_optimize += codec_params

        # VAE LoRA adapter params
        vae_lora_params = [
            p for n, p in self.vae.named_parameters() if "lora" in n and p.requires_grad
        ]

        # DiT LoRA adapter params
        DiT_lora_params = [
            p for n, p in self.DiT.named_parameters() if "lora" in n and p.requires_grad
        ]

        params_to_optimize += vae_lora_params
        params_to_optimize += DiT_lora_params
        
        prompt_params = [p for n, p in self.prompter.named_parameters() if p.requires_grad]
        params_to_optimize += prompt_params

        if self.use_vf is not None and self.vf_latent:
            # VF params
            vf_params = list(self.linear_proj.parameters())
            params_to_optimize += vf_params

        return params_to_optimize
    
    def get_state_dicts_for_save(self):
        save_dict = {}

        # Latent Codec
        codec = getattr(self.codec, "module", self.codec)
        save_dict["state_dict_codec"] = codec.state_dict()
        
        vae = getattr(self.vae, "module", self.vae)
        DiT = getattr(self.DiT, "module", self.DiT)

        # VAE LoRA adapter
        vae_lora_state = {
            k: v for k, v in vae.state_dict().items() if "lora" in k
        }
        save_dict["state_dict_vae"] = vae_lora_state

        # DiT LoRA adapter
        DiT_lora_state = {
            k: v for k, v in DiT.state_dict().items() if "lora" in k
        }
        save_dict["state_dict_DiT"] = {**DiT_lora_state}

        prompter = getattr(self.prompter, "module", self.prompter)
        save_dict["state_dict_prompter"] = prompter.state_dict()

        if self.use_vf is not None and self.vf_latent:
            # VF params
            vf_state = list(self.linear_proj.parameters())
            save_dict["state_dict_vf"] = vf_state

        return save_dict
    
    def get_last_layer(self):
        return self.codec.aux.block[-1].weight
    
    def get_vf_last_layer(self):
        return self.codec.aux.block[-1].weight
    
    def forward(self, c_t, cfg=False):
        # 1. Encoder
        latent2 = self.aux_codec((c_t + 1) / 2).detach()
        lq_latent = self.vae.encode(c_t).latent * self.vae.config.scaling_factor


        # 2. Latent Codec
        # torch.cuda.synchronize()
        # start = time.time()
        
        out = self.codec(lq_latent, latent2) 
        lq_mean_hat = out["mean"]
        lq_log_scale_hat = out["scale"]
        lq_log_scale_hat = torch.clamp(lq_log_scale_hat, -30.0, 20.0)
        lq_scale_hat = torch.exp(0.5 * lq_log_scale_hat)
        lq_scale_var = torch.exp(lq_log_scale_hat)
        
        res = out["res"]
        prompt = out["prompt"]
        rate = out["likelihoods"]
        
        if self.codec_mode == 'self_dist':
            lq_latent_hat = lq_mean_hat
        elif self.codec_mode == 'sample':
            sample = randn_tensor(
                lq_mean_hat.shape, generator=None, device=lq_mean_hat.device, dtype=lq_mean_hat.dtype
            )
            lq_latent_hat = lq_mean_hat + lq_scale_hat * sample
            kl_loss = 0.5 * torch.sum(
                    torch.pow(lq_mean_hat, 2) + lq_scale_var - 1.0 - lq_log_scale_hat,
                    dim=[1, 2, 3],
                )
        elif self.codec_mode == 'kl_mean':
            lq_latent_hat = lq_mean_hat
            kl_loss = 0.5 * torch.sum(
                    torch.pow(lq_mean_hat, 2) + lq_scale_var - 1.0 - lq_log_scale_hat,
                    dim=[1, 2, 3],
                )
        
        # 3. Encode input prompt
        pos_caption_enc = self.prompter(prompt)

        # 4. Denoising loop
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = self.sched.base_scheduler.timesteps
        timestep = t.expand(lq_latent_hat.shape[0])
        timestep = timestep * self.DiT.config.timestep_scale

        # predict noise model_output
        model_pred = self.DiT(
            lq_latent_hat,
            encoder_hidden_states=pos_caption_enc,
            # encoder_attention_mask=prompt_attention_mask,
            timestep=timestep,
            return_dict=False,
            # attention_kwargs=self.attention_kwargs,
        )[0]

        # compute previous image: x_t -> x_t-1
        x_denoised = self.sched.step(model_pred, lq_scale_hat, self.timesteps, lq_latent_hat, return_dict=True) + res

        if self.codec_mode == 'self_dist':
            # kl_loss = 0.05 * (1 - F.cosine_similarity(x_denoised, lq_latent)).mean()
            kl_loss = 0.05 * F.relu(1 - 0.5 - F.cosine_similarity(x_denoised, lq_latent)).mean()

        # 5. Decoder
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

        # torch.cuda.synchronize()
        # end = time.time()
        
        # avg_time = (end - start)
        # print(f"===========Average forward time per iteration=========: {avg_time * 1000:.2f} ms")

        # VF output:
        if self.use_vf is not None:
            aux_feature = self.foundation_model(c_t, False)
            if self.vf_latent:
                if not self.reverse_proj:
                    aux_feature = self.linear_proj(aux_feature)
                else:
                    z = self.linear_proj(lq_latent_hat)
            else:
                z = self.foundation_model(output_image, True)
                
            return output_image, rate, z, aux_feature, kl_loss
                
        return output_image, rate, None, None, kl_loss
    
    def compress(self, c_t):
        
        # 1. Encoder
        latent2 = self.aux_codec((c_t + 1) / 2).detach()
        lq_latent = self.vae.encode(c_t).latent * self.vae.config.scaling_factor

        # 2. Latent Codec
        output_dict = self.codec.compress(lq_latent, latent2)

        return output_dict
    
    def decompress(self, strings, shape, cfg=False):

        # 1. Latent Codec - Entropy Decoding
        lq_mean_hat, lq_log_scale_hat, res, prompt = self.codec.decompress(strings, shape)
        lq_log_scale_hat = torch.clamp(lq_log_scale_hat, -30.0, 20.0)
        lq_scale_hat = torch.exp(0.5 * lq_log_scale_hat)
        lq_scale_var = torch.exp(lq_log_scale_hat)

        if self.codec_mode == 'sample':
            sample = randn_tensor(
                lq_mean_hat.shape, generator=None, device=lq_mean_hat.device, dtype=lq_mean_hat.dtype
            )
            lq_latent_hat = lq_mean_hat + 0.1 * lq_scale_hat * sample
        else: 
            lq_latent_hat = lq_mean_hat
        
        # 2. Encode input prompt
        pos_caption_enc = self.prompter(prompt)

        # 3. Denoising loop
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = self.sched.base_scheduler.timesteps
        timestep = t.expand(lq_latent_hat.shape[0])
        timestep = timestep * self.DiT.config.timestep_scale

        # predict noise model_output
        model_pred = self.DiT(
            lq_latent_hat,
            encoder_hidden_states=pos_caption_enc.to(lq_latent_hat.device),
            # encoder_attention_mask=prompt_attention_mask,
            timestep=timestep.to(lq_latent_hat.device),
            return_dict=False,
            # attention_kwargs=self.attention_kwargs,
        )[0]

        # compute previous image: x_t -> x_t-1
        x_denoised = self.sched.step(model_pred, lq_scale_hat, self.timesteps, lq_latent_hat, return_dict=True) + res

        # Decoder
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

        return output_image
    
    def _set_latent_tile(self, latent_tiled_size = 96, latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap
    
