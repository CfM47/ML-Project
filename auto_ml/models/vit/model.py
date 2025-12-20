import torch
import torch.nn as nn

class ViTSegmentation(nn.Module):
    def __init__(self, 
                 image_size=512, 
                 patch_size=16, 
                 num_classes=3, 
                 dim=768, 
                 depth=12, 
                 heads=12, 
                 mlp_dim=3072, 
                 channels=1, 
                 dropout=0.1, 
                 emb_dropout=0.1):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.dim = dim

        # Patch Embedding using Conv2d
        # This is more robust for MPS/CUDA backward passes than Rearrange+View
        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- CNN Decoder (Progressive Upsampling) ---
        # Feature map size: (H/P, W/P) -> (512/16, 512/16) = (32, 32)
        
        self.decoder = nn.Sequential(
            # Block 1: 32x32 -> 64x64
            nn.Conv2d(dim, 512, kernel_size=3, padding=1),
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

    def forward(self, img):
        # img: (B, C, H, W)
        
        # 1. Patch Embedding
        # (B, C, H, W) -> (B, Dim, H/P, W/P)
        x = self.patch_embed(img)
        
        # Flatten: (B, Dim, H/P, W/P) -> (B, Dim, NumPatches)
        x = x.flatten(2)
        
        # Transpose: (B, Dim, NumPatches) -> (B, NumPatches, Dim)
        x = x.transpose(1, 2).contiguous()
        
        # 2. Add Positional Embedding
        x += self.pos_embedding
        x = self.dropout(x)
        
        # 3. Transformer Encoder
        x = self.transformer(x) # (B, NumPatches, Dim)
        
        # 4. Reshape for CNN Decoder
        # (B, NumPatches, Dim) -> (B, Dim, NumPatches)
        x = x.transpose(1, 2).contiguous()
        
        # (B, Dim, NumPatches) -> (B, Dim, H/P, W/P)
        h = w = self.image_size // self.patch_size # 32
        x = x.reshape(x.shape[0], self.dim, h, w)
        
        # 5. Decode
        x = self.decoder(x) # (B, NumClasses, ImageSize, ImageSize)
        
        return x
