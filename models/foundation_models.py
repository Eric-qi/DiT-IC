import timm
import math
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model



def create_foundation_model(type, device='cuda'):
    assert type in ['mae', 'dinov2', 'clip'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024
    elif type == 'clip':
        return get_clip_encoder(device), 1024
    
class get_clip_encoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        """
        Load the CLIP pretrained ViT-L encoder from the timm library.
        """
        self.model, _ = clip.load("ViT-L/14", device=device)
        self.device = device
        self.normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.normalize(x)

        return self.model.encode_image(x)


class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type, device):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type, device)
        self.type = type
        self.feature_dim = feature_dim

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_clip(self, x):
        return self.model(x)
        
    # def forward_dinov2(self, x):
    #     b, c, h, w = x.shape # 512x512
    #     x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
    #     return self.model.forward_features(x)[:, 1:].reshape(b, h//8, w//8, -1).permute(0, 3, 1, 2)
        
    def forward(self, x, is_rec):
        if is_rec:
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)
            elif self.type == 'clip':
                return self.forward_clip(x)
        else:
            with torch.no_grad():
                if self.type == 'mae':
                    return self.forward_mae(x)
                elif self.type == 'dinov2':
                    return self.forward_dinov2(x)
                elif self.type == 'clip':
                    return self.forward_clip(x)
            
            