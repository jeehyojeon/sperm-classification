import torch
import argparse
import os
from models.model import SpermNormalityModel
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

def load_model(weight_path, backbone='densenet121'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpermNormalityModel(backbone_type=backbone).to(device)
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
        print(f"Loaded weights from {weight_path}")
    model.eval()
    return model, device

def run_inference(model, device, image_path):
    # Preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        det_out, cls_out = model(img_tensor)
        
    normality_prob = torch.sigmoid(cls_out).item()
    # Optimized threshold from the manuscript
    threshold = 0.4271
    status = "Normal" if normality_prob >= threshold else "Abnormal"
    
    print(f"Image: {image_path}")
    print(f"Normality Probability: {normality_prob:.4f}")
    print(f"Classification: {status}")
    
    return normality_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymous Sperm Normality Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/best_model.pt", help="Path to model weights")
    args = parser.parse_args()
    
    model, device = load_model(args.weights)
    run_inference(model, device, args.image)
