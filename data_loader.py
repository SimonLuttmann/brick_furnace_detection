import numpy as np
import tifffile
from sklearn.model_selection import train_test_split
import os

def load_all_images(path: str, load_into_ram: bool = True):
    
    full_root_path = os.path.abspath(path)
    if os.path.exists(full_root_path):
        print(f"Root path found: {full_root_path}")
    else:
        print(f"Root path not found: {full_root_path}")
        raise FileNotFoundError(f"Root path does not exist: {full_root_path}")
    
    all_images = []
    all_image_paths = []
    all_labels = []
    all_labels_paths = []
    
    image_dir = os.path.join(full_root_path, "Image")
    label_dir = os.path.join(full_root_path, "Mask")
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            label_path = os.path.join(label_dir, file)


            if (load_into_ram):
                img = tifffile.imread(image_path)
                all_images.append(img)
                label = tifffile.imread(label_path)
                all_labels.append(label)


            all_image_paths.append(image_path)
            all_labels_paths.append(label_path)
            
    return all_images, all_image_paths, all_labels, all_labels_paths

def load_split(path: str, load_into_ram: bool = True, verbose: bool = True):
    images, image_paths, labels, label_paths = load_all_images(path, load_into_ram=load_into_ram)
    if (load_into_ram):
        X_train, X_test, y_train, y_test, image_paths_train, image_paths_test, label_paths_train, label_paths_test = train_test_split(images, labels, image_paths, label_paths, test_size=0.2, random_state=42) 
        if (verbose):            
            print(f"X_train size: {len(X_train)}")
            print(f"X_test size: {len(X_test)}")
            print(f"Image paths: {f'training image paths: {image_paths_train[:2]}'}")
            print(f"Label paths: {f'training label paths: {label_paths_train[:2]}'}")
            print(f"Image paths test: {f'test image paths: {image_paths_test[:2]}'}")
            print(f"Label paths test: {f'test label paths: {label_paths_test[:2]}'}")
        return X_train, X_test, y_train, y_test, image_paths_train, image_paths_test, label_paths_train, label_paths_test
    else:        
        image_paths_train, image_paths_test, label_paths_train, label_paths_test = train_test_split(image_paths, label_paths, test_size=0.2, random_state=42) 
        if (verbose):
            print(f"Image paths: {f'training image paths: {image_paths_train[:2]}'}")
            print(f"Label paths: {f'training label paths: {label_paths_train[:2]}'}")
            print(f"Image paths test: {f'test image paths: {image_paths_test[:2]}'}")
            print(f"Label paths test: {f'test label paths: {label_paths_test[:2]}'}")
        return None, None, None, None, image_paths_train, image_paths_test, label_paths_train, label_paths_test
    

    
if __name__ == "__main__":
    example_path = '/home/simon/Documents/studium/deep_learning/case_study/brick_furnace_detection/data/Brick_Data_Train/Brick_Data_Train'
    X_train, X_test, y_train, y_test, X_train_path, X_test_path, y_train_path, y_test_path = load_split(example_path, load_into_ram=True)
