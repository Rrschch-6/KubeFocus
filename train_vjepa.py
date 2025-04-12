import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import tqdm

# Assuming these are defined in separate modules
from model import SimplifiedVisionTransformer, SimplePredictor
from dataset import VideoDataset
from util import generate_masks
#TODO: Add tensorboard
# from tensorboardX import SummaryWriter

def main(args, device):
    # Initialize models
    encoder = SimplifiedVisionTransformer(
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        depth=1,
        num_heads=2,
        mlp_ratio=4.0
    ).to(device)

    predictor = SimplePredictor(
        embed_dim=args.embed_dim,
        predictor_embed_dim=32,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        img_size=args.img_size,
        use_mask_tokens=True
    ).to(device)

    target_encoder = copy.deepcopy(encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate
    )

    # Dataset and DataLoader
    # Replace with actual video file paths
    dataset = VideoDataset(args.video_paths, frames_per_video=args.num_frames, train=True)
    val_dataset = VideoDataset(args.video_paths, frames_per_video=args.num_frames, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    def reg_fn(z) -> torch.Tensor:
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
    

    # Training loop
    pbar = tqdm.trange(0, args.num_epochs, desc=f'{args.num_frames} frames per video')
    val_loss = torch.tensor(0.0)
    for epoch in pbar:
        total_loss = 0
        encoder.train()
        target_encoder.train()
        predictor.train()
        for itr, x in enumerate(loader):
            x = x.to(device)  # Shape: (B, 3, num_frames, 32, 32)
            B = x.size(0)

            # Generate masks
            masks_enc, masks_pred = generate_masks(B, args.total_patches, args.mask_ratio)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)

            # Forward target encoder (full input)
            with torch.no_grad():
                h_full = target_encoder(x)  # (B, total_patches, embed_dim)
                # Select features for masked patches
                num_pred = args.total_patches - int(args.total_patches * (1 - args.mask_ratio))
                h = h_full[masks_pred].view(B, num_pred, args.embed_dim)  # (B, num_pred, embed_dim)

            # Forward encoder (visible patches)
            z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

            # Forward predictor (predict masked patches)
            pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

            # Compute loss
            loss_jepa = F.mse_loss(pred, h)
            pstd_z = reg_fn(z)  # predictor variance across patches
            loss_reg = torch.mean(F.relu(1.-pstd_z))
            loss = loss_jepa + args.reg_coeff * loss_reg

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target encoder with EMA
            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data = args.momentum * param_k.data + (1 - args.momentum) * param_q.data

            total_loss += loss.item()

            # Logging every 10 iterations
            if itr % 10 == 0:
                pbar.set_postfix({"Iter": f"{itr}/{len(loader)}", "Loss": f"{loss.item():.4f}", "Val Loss": f"{val_loss.item():.4f}"})

        # Average loss per epoch
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            # Save checkpoint
            checkpoint = {
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, f'logs/checkpoint_{args.num_frames}_fpv_epoch_{epoch+1}.pth')

            @torch.no_grad()
            def validate() -> None:
                print("Validating...")
                target_encoder.eval()
                encoder.eval()
                val_loss = 0
                for x in val_loader:
                    x = x.to(device)  # Shape: (B, 3, num_frames, 32, 32)
                    B = x.size(0)

                    # Generate masks
                    masks_enc, masks_pred = generate_masks(B, args.total_patches, args.mask_ratio)
                    masks_enc = masks_enc.to(device)
                    masks_pred = masks_pred.to(device)

                    # Forward target encoder (full input)
                    with torch.no_grad():
                        h_full = target_encoder(x)  # (B, total_patches, embed_dim)
                        # Select features for masked patches
                        num_pred = args.total_patches - int(args.total_patches * (1 - args.mask_ratio))
                        h = h_full[masks_pred].view(B, num_pred, args.embed_dim)  # (B, num_pred, embed_dim)

                    # Forward encoder (visible patches)
                    z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

                    # Forward predictor (predict masked patches)
                    pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

                    # Compute loss
                    val_loss += F.mse_loss(pred.detach(), h.detach())
                
                val_loss /= len(val_loader)
                print(f"Validation Loss Average: {val_loss.item():.4f}")
                pbar.set_postfix({"Iter": f"{itr}/{len(loader)}", "Loss": f"{loss.item():.4f}", "Val Loss": f"{val_loss.item():.4f}"})
                return val_loss 

            # Call validation function
            val_loss = validate()              


    print("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VJEPa model")
    parser.add_argument("--video_paths", type=str, required=True, help="Path to video dataset")
    parser.add_argument("--num_frames", type=int, required=True, help="Number of frames per video")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for patches")
    parser.add_argument("--momentum", type=float, default=0.999, help="Momentum for target encoder update")
    parser.add_argument("--reg_coeff", type=float, default=0.01, help="Regularization coefficient")
    parser.add_argument("--total_patches", type=int, default=80, help="Total patches per video")
    parser.add_argument("--img_size", type=int, default=32, help="Image size (height and width)")

    args = parser.parse_args()
    
    # Training
    args.total_patches = args.num_frames * (args.img_size // args.patch_size) ** 2
    print(f"Total patches: {args.total_patches}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device=device)
