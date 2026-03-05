import os
import csv
import numpy as np
from PIL import Image

def generate_dummy_data(root_dir='dataset', num_samples=5):
    """
    Generates dummy images and CSV labels matching the repository's expected format.
    """
    subsets = ['train', 'val', 'test']
    
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        img_dir = os.path.join(subset_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        csv_path = os.path.join(subset_dir, 'obb_labels.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'class'])
            
            for i in range(num_samples):
                # Random noise image
                img_name = f'dummy_{i}.png'
                img_path = os.path.join(img_dir, img_name)
                
                # Create a 224x224 RGB image with random colors
                img_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_data)
                img.save(img_path)
                
                # Assign 0 (abnormal) or 1 (normal)
                # Following bitwise weights logic, strength (0-5)
                strength = np.random.randint(0, 6)
                writer.writerow([img_name, strength])
                
    print(f"Successfully generated dummy dataset at '{root_dir}/' with {num_samples} samples per subset.")

if __name__ == "__main__":
    generate_dummy_data()
