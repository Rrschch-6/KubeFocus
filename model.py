import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SimpleDetectorHead(nn.Module):
    """A simple detector using the pre-trained simplified V-JEPA 
    trained on our dataset as backbone for intrusion detection."""
    def __init__(self, input_dim=64, num_classes=1, img_size=32, patch_size=8):
        super(SimpleDetectorHead, self).__init__()
        self.num_patches = (img_size // patch_size)**2  # e.g., 32 / 8 **2 = 16
        self.fc = nn.Linear(input_dim*self.num_patches, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten the input
        x = self.fc(x)
        return x.squeeze(-1)  # Remove the last dimension
    
class SimplePredictor(nn.Module):
    """A lightweight predictor for 32x32x3xnum_frames input, avoiding transformers."""
    def __init__(
        self,
        embed_dim=768,           # Dimension of input embeddings
        predictor_embed_dim=128, # Lower dimension for predictor processing
        num_frames=1,            # Number of frames in the input
        patch_size=8,            # Size of each patch (e.g., 8x8)
        img_size=32,             # Image size (32x32)
        in_channels=3,          # Number of input channels (e.g., RGB)
        use_mask_tokens=False,   # Whether to use learnable mask tokens
        init_std=0.02            # Standard deviation for initialization
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels

        # Calculate number of patches
        grid_size = img_size // patch_size  # e.g., 32 / 8 = 4
        self.num_patches_per_frame = grid_size * grid_size  # e.g., 4x4 = 16
        self.total_patches = self.num_patches_per_frame * num_frames

        # Project input embeddings to a lower dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        # Optional mask tokens for masked patches
        self.mask_tokens = None
        if use_mask_tokens:
            self.mask_tokens = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            torch.nn.init.trunc_normal_(self.mask_tokens, std=init_std)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches, predictor_embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=init_std)

        # Simple MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim)
        )

        # Project back to original embedding dimension
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=0):
        """
        Forward pass to predict target tokens from context tokens.

        :param ctxt: Context tokens (B, N_ctxt, embed_dim)
        :param tgt: Target tokens (B, N_tgt, embed_dim)
        :param masks_ctxt: List of context masks (B, total_patches)
        :param masks_tgt: List of target masks (B, total_patches)
        :param mask_index: Index for mask token (if used, ignored here as num_mask_tokens=1)
        :return: Predicted target tokens (B, N_tgt, embed_dim)
        """
        # Ensure masks are provided and in list form
        assert masks_ctxt is not None and masks_tgt is not None, "Masks are required"
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        B = ctxt.size(0)  # Batch size

        # Project context tokens to predictor dimension
        x = self.predictor_embed(ctxt)  # (B, N_ctxt, predictor_embed_dim)

        # Add positional embeddings to context tokens
        pos_emb_ctxt = self.pos_embed[:, :x.size(1), :]
        x = x + pos_emb_ctxt

        # Handle target tokens
        if self.mask_tokens is not None:
            # Use learnable mask tokens for target positions
            num_tgt = self.total_patches - x.size(1)
            pred_tokens = self.mask_tokens.expand(B, num_tgt, -1)
        else:
            # Project target tokens and add small noise (simplified diffusion)
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = pred_tokens + torch.randn_like(pred_tokens) * 0.1

        # Add positional embeddings to target tokens
        pos_emb_tgt = self.pos_embed[:, x.size(1):, :]
        pred_tokens = pred_tokens + pos_emb_tgt

        # Concatenate context and target tokens
        x = torch.cat([x, pred_tokens], dim=1)  # (B, total_patches, predictor_embed_dim)

        # Pass through MLP
        x = self.mlp(x)

        # Project back to original embedding dimension
        x = self.predictor_proj(x)

        # Return predictions for target tokens only
        N_ctxt = ctxt.size(1)
        return x[:, N_ctxt:, :]

class SimplifiedVisionTransformer(nn.Module):
    def __init__(
        self,
        num_frames,
        embed_dim=64,
        patch_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        in_channels=3,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Patch embedding: 2D convolution applied per frame (tubelet_size=1)
        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Calculate number of patches
        grid_size = 32 // patch_size  # 32 / 8 = 4
        self.num_patches_per_frame = grid_size * grid_size  # 4 * 4 = 16
        self.total_patches = self.num_patches_per_frame * num_frames

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])

        # Normalization layer
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        """
        Args:
            x (torch.Tensor): Input video of shape (B, C, T, H, W) = (batch_size, 3, num_frames, 32, 32)
            masks (torch.Tensor, optional): Boolean tensor of shape (B, total_patches) indicating visible patches.
                                           Assumes each sample has the same number of True values if provided.
        Returns:
            torch.Tensor: Encoded output of shape (B, num_visible_patches, embed_dim) if masks are provided,
                          or (B, total_patches, embed_dim) if not.
        """
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Patch embedding: reshape to (B*T, C, H, W) and apply 2D convolution
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, embed_dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (B*T, 16, embed_dim)
        x = x.reshape(B, T * self.num_patches_per_frame, self.embed_dim)  # (B, total_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply masking if provided
        if masks is not None:
            num_visible = masks.sum(dim=1)  # Number of visible patches per batch, shape (B,)
            assert (num_visible == num_visible[0]).all(), "All batches must have the same number of visible patches"
            x = x[masks].view(B, -1, self.embed_dim)  # (B, num_visible, embed_dim)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply normalization
        x = self.norm(x)
        return x
    

# Example usage
if __name__ == "__main__":
    num_frames = 5
    in_channels = 3
    model = SimplifiedVisionTransformer(num_frames=num_frames, in_channels=in_channels)
    x = torch.randn(2, in_channels, num_frames, 32, 32)
    total_patches = model.total_patches  # 5 * 16 = 80
    masks = torch.zeros(2, total_patches, dtype=torch.bool)
    masks[:, :40] = True  # Keep 40 patches per sample
    y = model(x, masks)
    print(f"Output shape: {y.shape}")  # Should be (2, 40, 64)


    predictor = SimplePredictor(
        embed_dim=64,
        predictor_embed_dim=32,
        num_frames=5,
        patch_size=8,
        img_size=32
    )
    # Dummy inputs
    ctxt = torch.randn(2, 40, 64)  # 40 context patches
    tgt = torch.randn(2, 40, 64)   # 40 target patches
    masks_ctxt = torch.zeros(2, 80).bool()  # 80 total patches, context unmasked
    masks_tgt = torch.ones(2, 80).bool()    # target masked
    pred = predictor(ctxt, tgt, masks_ctxt, masks_tgt)
    print(f"Output shape: {pred.shape}")  # Expected: (2, 40, 64)