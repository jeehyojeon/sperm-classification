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

    # Mock DataLoaders (to be replaced with real data loading logic as per dataset README)
    # train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    logger.info("Model initialized. Starting training loop...")

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        # for batch in train_loader:
        #     images, labels, strengths = batch
        #     optimizer.zero_grad()
        #     det_out, cls_logits = model(images)
        #     loss = criterion(cls_logits, labels, strengths)
        #     loss.backward()
        #     optimizer.step()
        
        # Validation logic here...
        model.eval()
        # val_loss = ... (calculation)
        val_loss = 1.0 / (epoch + 1) # Placeholder logic minimizing loss
        
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
