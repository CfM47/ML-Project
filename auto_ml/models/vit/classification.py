import torch
import torch.nn as nn


class ViTClassification(nn.Module):
    """
    Vision Transformer Classifier for image classification.

    Uses a CLS token approach where a learnable classification token is
    prepended to the patch sequence. The transformer processes all tokens,
    and the final CLS token representation is used for classification.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 3,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        channels: int = 1,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        """
        Initialize the ViT Classifier.

        Args:
            image_size: Input image size (must be divisible by patch_size).
            patch_size: Size of each patch.
            num_classes: Number of output classes.
            dim: Transformer embedding dimension.
            depth: Number of transformer encoder layers.
            heads: Number of attention heads.
            mlp_dim: Dimension of the MLP feedforward layer.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            dropout: Dropout rate in transformer and classification head.
            emb_dropout: Dropout rate after positional embedding.

        """
        super().__init__()

        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.dim = dim

        # Patch Embedding using Conv2d
        # This is more robust for MPS/CUDA backward passes than Rearrange+View
        self.patch_embed = nn.Conv2d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding: +1 for CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

        # Softmax for probability distribution output
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the ViT Classifier.

        Args:
            x: Input tensor of shape (B, C, H, W). H and W must equal image_size.
            return_logits: If True, return raw logits instead of probabilities.
                          Use True when training with CrossEntropyLoss.

        Returns:
            Tensor of shape (B, num_classes) with probabilities (or logits if
            return_logits=True). Probabilities sum to 1 along dim=1.

        """
        batch_size = x.shape[0]

        # 1. Patch Embedding
        # (B, C, H, W) -> (B, Dim, H/P, W/P)
        x = self.patch_embed(x)

        # Flatten: (B, Dim, H/P, W/P) -> (B, Dim, NumPatches)
        x = x.flatten(2)

        # Transpose: (B, Dim, NumPatches) -> (B, NumPatches, Dim)
        x = x.transpose(1, 2).contiguous()

        # 2. Prepend CLS token
        # Expand cls_token for batch: (1, 1, Dim) -> (B, 1, Dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate: (B, 1, Dim) + (B, NumPatches, Dim) -> (B, NumPatches+1, Dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add Positional Embedding
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 4. Transformer Encoder
        x = self.transformer(x)  # (B, NumPatches+1, Dim)

        # 5. Extract CLS token output
        cls_output = x[:, 0]  # (B, Dim)

        # 6. Classification
        x = self.classifier(cls_output)  # (B, num_classes)

        # Return probabilities or logits
        if return_logits:
            return x

        sm: torch.Tensor = self.softmax(x)

        return sm
