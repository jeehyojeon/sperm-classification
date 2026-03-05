import torch
import yaml
from models.model import SpermNormalityModel
from utils.metrics import compute_metrics
from utils.seed import set_seed
import numpy as np
import os

def test():
    # Load configuration
    with open('configs/miccai2026.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpermNormalityModel(backbone_type=config['backbone']).to(device)
    
    weight_path = os.path.join('weights', 'best_model.pt')
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('features.'):
                new_state_dict[k.replace('features.', 'backbone.')] = v
            elif k.startswith('cbam.'):
                new_state_dict[k.replace('cbam.', 'classification_head.cbam.')] = v
            elif k.startswith('classifier.'):
                new_state_dict[k.replace('classifier.', 'classification_head.classifier.')] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {weight_path} (Remapped keys, strict=False)")
    else:
        print("Warning: weights/best_model.pt not found. Using randomly initialized weights.")

    model.eval()

    # Initialize Test DataLoader with relative paths from config
    from utils.dataset import SpermDataset
    from torch.utils.data import DataLoader

    test_dataset = SpermDataset(
        image_dir=config.get('test_img', 'dataset/test/images'),
        csv_path=config.get('test_csv', 'dataset/test/obb_labels.csv'),
        img_size=224
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Loaded {len(test_dataset)} test samples.")
    print("\n--- Running Evaluation ---")
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels, _ = batch # strengths are not used for evaluation
            images, labels = images.to(device), labels.to(device)
            det_out, cls_logits = model(images)
            probs = torch.sigmoid(cls_logits.view(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())

    if len(all_probs) == 0:
        print("\nError: No samples found for evaluation. Please check your test dataset.")
        return

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Compute metrics using the utility function
    metrics = compute_metrics(all_probs, all_labels, threshold=config['normality_threshold'])

    print("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    test()
