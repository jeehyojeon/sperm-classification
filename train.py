import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import SpermNormalityModel
from losses.loss import get_loss_fn
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.metrics import compute_metrics
import numpy as np

def train():
    # Load configuration
    with open('configs/miccai2026.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    set_seed(config['seed'])

    # Setup logger
    save_dir = config.get('save_dir', 'runs/exp1')
    logger = setup_logger(save_dir)
    logger.info("Starting training with configuration: {}".format(config))

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpermNormalityModel(
        backbone_type=config['backbone'],
        pretrained=True
    ).to(device)

    # Load pre-existing weights if available
    weight_path = os.path.join('weights', 'best_model.pt')
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info(f"Loaded weights from {weight_path}")
    model.eval() # This line was explicitly requested, though typically model.train() would be called before the training loop.

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize loss function
    criterion = get_loss_fn(config)

    # Initialize DataLoaders with relative paths from config
    from utils.dataset import SpermDataset
    
    train_dataset = SpermDataset(
        image_dir=config.get('train_img', 'dataset/train/images'),
        csv_path=config.get('train_csv', 'dataset/train/obb_labels.csv'),
        img_size=224
    )
    val_dataset = SpermDataset(
        image_dir=config.get('val_img', 'dataset/val/images'),
        csv_path=config.get('val_csv', 'dataset/val/obb_labels.csv'),
        img_size=224
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    logger.info("Model initialized. Starting training loop...")

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            det_out, cls_logits = model(images)
            # Use raw logits for BCEWithLogitsLoss / FocalLoss
            loss = criterion(cls_logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
        
        # Validation logic
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                det_out, cls_logits = model(images)
                val_loss += criterion(cls_logits.squeeze(), labels).item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('weights', 'best_model.pt'))
            logger.info(f"New best model saved with Loss: {best_val_loss:.4f}")

        scheduler.step()

    logger.info("Training complete.")

if __name__ == "__main__":
    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)
    train()
