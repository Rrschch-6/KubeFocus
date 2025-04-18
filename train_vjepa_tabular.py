from dataset import TabularDataset  # Note: The query mentions TabularData, but we'll assume TabularDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import tqdm

# Assuming these are defined in separate modules
from model import TabularTransformerEncoder, TabularPredictor
from dataset import TabularDataset
from util import generate_row_masks

def main(args, device):
    # Initialize models
    encoder = TabularTransformerEncoder(
        num_features=args.num_features,
        max_length=args.max_length,
        embed_dim=args.embed_dim,
        num_heads=2,
        depth=1
    ).to(device)

    predictor = TabularPredictor(
        embed_dim=args.embed_dim,
        max_len=args.max_length,
        predictor_embed_dim=32
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
    dataset = TabularDataset(args.data_path, max_len=args.max_length, label_col_name=args.label_col_name)
    val_dataset = TabularDataset(args.val_data_path, max_len=args.max_length, label_col_name=args.label_col_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    def reg_fn(z):
        """Regularization function to encourage variance in representations."""
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

    # Training loop
    pbar = tqdm.trange(0, args.num_epochs, desc=f'Tabular data with max length {args.max_length}')
    val_loss = torch.tensor(0.0)
    for epoch in pbar:
        total_loss = 0
        encoder.train()
        target_encoder.train()
        predictor.train()
        for itr, data in enumerate(loader):
            x = data['sequence'].to(device)  # Shape: (B, max_length, num_features)
            B = x.size(0)

            # Generate masks for rows
            masks_enc, masks_pred = generate_row_masks(B, args.max_length, args.mask_ratio)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)

            # Forward target encoder (full input)
            with torch.no_grad():
                h_full = target_encoder(x)  # (B, max_length, embed_dim)
                num_pred = max(1, args.max_length - int(args.max_length * (1 - args.mask_ratio)))
                h = h_full[masks_pred].view(B, num_pred, args.embed_dim)  # (B, num_pred, embed_dim)

            # Forward encoder (visible rows)
            z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

            # Forward predictor (predict masked rows)
            pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

            # Compute loss
            loss_jepa = F.mse_loss(pred, h)
            pstd_z = reg_fn(z)  # Variance across rows
            loss_reg = torch.mean(F.relu(1. - pstd_z))
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
            torch.save(checkpoint, f'logs/checkpoint_tabular_{args.max_length}len_epoch_{epoch+1}.pth')

            @torch.no_grad()
            def validate():
                print("Validating...")
                target_encoder.eval()
                encoder.eval()
                predictor.eval()
                val_loss = torch.tensor(0).to(torch.float32)
                for data in val_loader:
                    x = data['sequence'].to(device)  # Shape: (B, max_length, num_features)
                    B = x.size(0)

                    # Generate masks
                    masks_enc, masks_pred = generate_row_masks(B, args.max_length, args.mask_ratio)
                    masks_enc = masks_enc.to(device)
                    masks_pred = masks_pred.to(device)

                    # Forward target encoder (full input)
                    with torch.no_grad():
                        h_full = target_encoder(x)  # (B, max_length, embed_dim)
                        num_pred = max(1, args.max_length - int(args.max_length * (1 - args.mask_ratio)))
                        h = h_full[masks_pred].view(B, num_pred, args.embed_dim)  # (B, num_pred, embed_dim)

                    # Forward encoder (visible rows)
                    z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

                    # Forward predictor (predict masked rows)
                    pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

                    # Compute loss
                    val_loss += F.mse_loss(pred, h).item()

                val_loss /= len(val_loader)
                print(f"Validation Loss Average: {val_loss:.4f}")
                pbar.set_postfix({"Iter": f"{itr}/{len(loader)}", "Loss": f"{loss.item():.4f}", "Val Loss": f"{val_loss:.4f}"})
                return val_loss

            # Call validation function
            val_loss = validate()

    print("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train V-JEPA model for tabular data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--label_col_name", type=str, default='attack', help="Name of the label column")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--num_features", type=int, required=True, help="Number of features per row")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Mask ratio for rows")
    parser.add_argument("--momentum", type=float, default=0.999, help="Momentum for target encoder update")
    parser.add_argument("--reg_coeff", type=float, default=0.01, help="Regularization coefficient")

    args = parser.parse_args()

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device=device)