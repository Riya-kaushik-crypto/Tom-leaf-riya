import os
import shutil

def fix_dataset_structure():
    """Fix common dataset structure issues"""
    
    # Check if we're in the right directory
    if not os.path.exists('dataset'):
        print("Creating dataset directory...")
        os.makedirs('dataset', exist_ok=True)
    
    # Check if train folder exists
    if not os.path.exists('dataset/train'):
        print("Creating train directory...")
        os.makedirs('dataset/train', exist_ok=True)
    
    # Check for class folders
    required_classes = ['early_blight', 'late_blight', 'healthy']
    for class_name in required_classes:
        class_path = os.path.join('dataset/train', class_name)
        if not os.path.exists(class_path):
            print(f"Creating {class_name} folder...")
            os.makedirs(class_path, exist_ok=True)
    
    print("\n✅ Dataset structure fixed!")
    print("\nNow place your PlantVillage images in:")
    print("  dataset/train/early_blight/")
    print("  dataset/train/late_blight/")
    print("  dataset/train/healthy/")

if __name__ == "__main__":
    fix_dataset_structure()