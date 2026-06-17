import os
from collections import defaultdict

def verify_dataset():
    """Verify your local PlantVillage dataset"""
    
    dataset_path = 'dataset/train'
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    print("=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"\nFound {len(classes)} classes:")
    
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if any(f.endswith(ext) for ext in image_extensions)]
        
        print(f"  📁 {class_name}: {len(images)} images")
        total_images += len(images)
        
        # Show sample images
        if len(images) > 0:
            print(f"     Sample: {images[0]}")
    
    print(f"\n📊 Total images: {total_images}")
    
    if total_images == 0:
        print("\n❌ No images found!")
        return False
    
    # Check for minimum images
    min_required = 10
    problematic = []
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if any(f.endswith(ext) for ext in image_extensions)]
        if len(images) < min_required:
            problematic.append(f"{class_name} (only {len(images)} images)")
    
    if problematic:
        print(f"\n⚠️  Warning: Some classes have less than {min_required} images:")
        for p in problematic:
            print(f"   - {p}")
    else:
        print(f"\n✅ Dataset looks good! Ready for training.")
    
    return True

if __name__ == "__main__":
    verify_dataset()