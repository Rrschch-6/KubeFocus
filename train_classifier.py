# Training script for Intrusion Detection using the pre-trained V-JEPA model on our dataset
import torch

from model import SimplifiedVisionTransformer, SimpleDetectorHead
from inference_video import SurpriseScoreEstimator
from dataset import DetectionDataset
from util import mask_future
from torch.utils.data import DataLoader

import copy
import os
import tqdm
import argparse


def main(args, device):
    """ 
    Trains the classifier head only, while using the pretrained encoder and predictors as backbones.
    1. Predicts the hidden embeddings using the pretrained encoder and predictor while freezing their weights
    2. feeds the embeddings to the classifier head
    3. computes the BCE loss to train the head only
    """
    # Initialize models
    cfg = SurpriseScoreEstimator.cfg()
    check_fpv(cfg, args.pretrained_model_path)
    
    (encoder,
     target_encoder,
     predictor) = SurpriseScoreEstimator.prepare_models(
        model_path=args.pretrained_model_path,
        cfg=cfg
    )
    encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)

    # Freeze encoder and predictor weights
    for param in encoder.parameters():
        param.requires_grad = False
    for param in predictor.parameters():
        param.requires_grad = False
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Classifier head
    classifier_head = SimpleDetectorHead(input_dim=cfg.embed_dim, 
                                         num_classes=1,
                                         img_size=cfg.img_size,
                                         patch_size=cfg.patch_size).to(device)
    classifier_head.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        classifier_head.parameters(),
        lr=args.learning_rate
    )
    # Dataset and DataLoader
    dataset = DetectionDataset(
        mix_dataset_path=args.mix_dataset_path,
        frames_per_video=cfg.num_frames,
        train=True
    )
    # val_dataset = DetectionDataset(
    #     benign_video_paths=args.mix_dataset_path,
    #     frames_per_video=args.num_frames,
    #     train=False
    # )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = None
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # Tensorboard writer
    # writer = SummaryWriter(log_dir=args.log_dir)
    
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    val_loss = torch.tensor(0.0)
    pbar = tqdm.trange(0, args.num_epochs, desc=f"Training classifier with fpv {cfg.num_frames}")
    for epoch in pbar:
        for data in loader:
            classifier_head.train()
            optimizer.zero_grad()
            x = data['video']
            labels = data['label']
            x = x.to(device).requires_grad_(False)  # Shape: (B, 3, num_frames, 32, 32)
            labels = labels.to(device).to(torch.torch.float32)
            B = x.size(0)

            # Generate masks
            masks_enc, masks_pred = mask_future(B, cfg.total_patches,
                                                cfg.mask_ratio, fpv=cfg.num_frames)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)

            # Forward target encoder (full input)
            with torch.no_grad():
                h_full = target_encoder(x)  # (B, total_patches, embed_dim)
                # Select features for masked patches
                num_pred = int(cfg.num_frames * cfg.mask_ratio) * int(cfg.total_patches / cfg.num_frames)
                h = h_full[masks_pred].view(B, num_pred, cfg.embed_dim)  # (B, num_pred, embed_dim)

                #NOTE: No need for grads here during classifier training
                # Forward encoder (visible patches)
                z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

                # Forward predictor (predict masked patches)
                pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

            # Forward pass through classifier head
            logits = classifier_head(pred)

            # Compute loss
            #TODO: Labels?
            loss = criterion(logits, labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})
            # writer.add_scalar('Loss/train', loss.item(), epoch)
        
        # Validation step at every epoch
        def validate():
            classifier_head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    x = x.to(device)  # Shape: (B, 3, num_frames, 32, 32)
                    B = x.size(0)

                    # Generate masks
                    masks_enc, masks_pred = mask_future(B, args.total_patches, 
                                                        args.mask_ratio, cfg.num_frames)
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

                    # Forward pass through classifier head
                    logits = classifier_head(pred)

                    # Compute loss
                    #TODO: Labels?
                    val_loss += criterion(logits, labels).item()

            return val_loss / len(val_loader)
        
        #TODO: Make val set to validate
        # val_loss = validate()
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
        # writer.add_scalar('Loss/val', val_loss, epoch)
        
        if (epoch % 10 == 0) or (epoch == args.num_epochs - 1):
            # Save checkpoint
            checkpoint = {
                'classifier_head': classifier_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': loss.item(),
                'config': cfg
            }
            os.makedirs(args.log_dir + '/detector_logs', exist_ok=True)
            torch.save(checkpoint, args.log_dir + f'/detector_logs/checkpoint_epoch_{epoch+1}.pth')
    # writer.close()

@torch.no_grad()
def inference_detection(model_path, classifier_path, device):
    """
    Load the encoder, and predictor weights, then loads classifier head weights
    and run inference on the train set calculating the accuracy, confusion matrix, 
    and F1 score.
    """
    # Load the encoder and predictor models
    cfg = SurpriseScoreEstimator.cfg()
    check_fpv(cfg, model_path)
    
    (encoder,
     target_encoder,
     predictor) = SurpriseScoreEstimator.prepare_models(
                                                model_path=model_path,
                                                cfg=cfg
                                            )
    encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)

    # Load the classifier head
    classifier_head = SimpleDetectorHead(input_dim=cfg.embed_dim, 
                                         num_classes=1,
                                         img_size=cfg.img_size,
                                         patch_size=cfg.patch_size).to(device)
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False) #NOTE: weights_only=False since we save cfg
    classifier_head.load_state_dict(checkpoint['classifier_head'])
    classifier_head.to(device)
    classifier_head.eval()
    # Freeze encoder and predictor weights
    for param in encoder.parameters():
        param.requires_grad = False
    for param in predictor.parameters():
        param.requires_grad = False
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Load the dataset
    dataset = DetectionDataset(
        mix_dataset_path=args.mix_dataset_path,
        frames_per_video=cfg.num_frames,
        train=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize metrics
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    all_logits = []
    # Inference loop
    pbar = tqdm.tqdm(loader, desc="Inference")
    for data in pbar:
        x = data['video']
        labels = data['label']
        x = x.to(device).requires_grad_(False)  # Shape: (B, 3, num_frames, 32, 32)
        labels = labels.to(device).to(torch.long)
        B = x.size(0)

        # Generate masks
        masks_enc, masks_pred = mask_future(B, cfg.total_patches,
                                            cfg.mask_ratio, fpv=cfg.num_frames)
        masks_enc = masks_enc.to(device)
        masks_pred = masks_pred.to(device)

        # Forward target encoder (full input)
        with torch.no_grad():
            h_full = target_encoder(x)  # (B, total_patches, embed_dim)
            # Select features for masked patches
            num_pred = int(cfg.num_frames * cfg.mask_ratio) * int(cfg.total_patches / cfg.num_frames)
            h = h_full[masks_pred].view(B, num_pred, cfg.embed_dim)  # (B, num_pred, embed_dim)

            # Forward encoder (visible patches)
            z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

            # Forward predictor (predict masked patches)
            pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

        # Forward pass through classifier head
        logits = classifier_head(pred)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu().to(torch.int8))
        all_preds.append(torch.sigmoid(logits).cpu().to(torch.int8))
        # Compute accuracy
        preds = torch.sigmoid(logits) > 0.5
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix({"accuracy": total_correct / total_samples})
    # Calculate final accuracy
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")
    # Calculate confusion matrix and F1 score
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_logits = torch.cat(all_logits).numpy()
    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    print(f"F1 Score: {f1:.4f}")
    

def check_fpv(cfg, pretrained_model_path):
    import re
    # Check the fpv
    match = re.search(r'checkpoint_(\d+)_fpv', pretrained_model_path)
    if match:
        fpv = int(match.group(1))
        cfg.num_frames = fpv
        cfg.total_patches = fpv * (cfg.img_size // cfg.patch_size) ** 2


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the classifier head")
    parser.add_argument("--mix_dataset_path", type=str, required=True, help="Path to the mix dataset")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--classifier_path", type=str, default=None, help="Path to the classifier head")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Mode: train or inference")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for Checkpoints and TensorBoard logs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        main(args, device)
    elif args.mode == "inference":
        assert args.classifier_path is not None, "Classifier path is required for inference.\
                                                    Give the --train flag for training to start"
        inference_detection(args.pretrained_model_path, 
                            args.classifier_path, 
                            device)
    else:
        raise ValueError("Invalid mode. Choose either 'train' or 'inference'.")

