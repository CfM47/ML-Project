import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer

class SwinSegmentation(nn.Module):
    def __init__(self, 
                 patch_size=[4, 4],
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7],
                 mlp_ratio=4.0,
                 dropout=0.0,
                 num_classes=3, 
                 channels=1):
        super().__init__()
        
        # 1. Swin Transformer Backbone (Customizable)
        self.swin = SwinTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Modify the first layer to accept 'channels' input (default 1 for grayscale)
        # Original: Conv2d(3, embed_dim, kernel_size=(4, 4), stride=(4, 4))
        # Note: self.swin.features[0][0] is the first patch embedding conv
        
        # We need to recreate it because we can't just change input channels if we want to be clean,
        # but technically we can just replace the module.
        original_first_layer = self.swin.features[0][0]
        self.swin.features[0][0] = nn.Conv2d(
            channels, 
            embed_dim, 
            kernel_size=original_first_layer.kernel_size, 
            stride=original_first_layer.stride
        )
        
        # Determine feature dimension automatically
        # Swin backbone final layer is 'head' (Linear), but we use features.
        # The output of features() is (B, H/32, W/32, last_dim)
        # last_dim is typically embed_dim * 2^(num_stages-1)
        # For Swin-T: 96 * 8 = 768
        self.dim = embed_dim * (2 ** (len(depths) - 1))
        
        # 2. Adapter (1/32 -> 1/16) to match ViT decoder input expectation
        # We need to upsample from H/32 to H/16
        self.adapter = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 3. CNN Decoder (Same as ViT)
        # Input: (B, Dim, H/16, W/16) -> (B, 32, 32) (if Image=512)
        
        self.decoder = nn.Sequential(
            # Block 1: 32x32 -> 64x64
            nn.Conv2d(self.dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Block 2: 64x64 -> 128x128
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Block 3: 128x128 -> 256x256
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Block 4: 256x256 -> 512x512
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Final Projection
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        
        # Extract features
        # Swin feature extractor returns (B, H/32, W/32, Dim)
        features = self.swin.features(x)
        
        # Permute to (B, Dim, H/32, W/32)
        features = features.permute(0, 3, 1, 2).contiguous()
        
        # Adapt to 1/16 resolution
        # (B, Dim, H/32, W/32) -> (B, Dim, H/16, W/16)
        x = self.adapter(features)
        
        # Decode
        x = self.decoder(x)
        
        return x
