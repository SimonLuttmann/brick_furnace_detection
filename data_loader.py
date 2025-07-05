import numpy as np
import tifffile
from sklearn.model_selection import train_test_split
from utils import print_available_memory
import os


def load_all_images(path: str, load_into_ram: bool = True, as_numpy: bool = False):
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

            if load_into_ram:
                img = tifffile.imread(image_path)
                all_images.append(img)
                label = tifffile.imread(label_path)
                all_labels.append(label)

            all_image_paths.append(image_path)
            all_labels_paths.append(label_path)

    # Convert to numpy arrays if requested
    if load_into_ram and as_numpy:
        all_images = np.stack(all_images, axis=0)
        all_labels = np.stack(all_labels, axis=0)
    if as_numpy:
        all_image_paths = np.array(all_image_paths)
        all_labels_paths = np.array(all_labels_paths)
    return all_images, all_image_paths, all_labels, all_labels_paths


def load_split(path: str, load_into_ram: bool = True, verbose: bool = True, as_numpy: bool = False):

    unused1, image_paths, unused2, label_paths = load_all_images(
        path, load_into_ram=False, as_numpy=False
    )
    image_paths_train, image_paths_test, label_paths_train, label_paths_test = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42
    )

    X_train, X_test, y_train, y_test = None, None, None, None

    if as_numpy and load_into_ram:
        # Reserve space for images: float32, shape (N, 256, 256, 8)
        X_train = np.empty((len(image_paths_train), 256, 256, 8), dtype=np.float32)
        X_test = np.empty((len(image_paths_test), 256, 256, 8), dtype=np.float32)
        # Reserve space for labels: uint8, shape (N, 256, 256, 1)
        y_train = np.empty((len(image_paths_train), 256, 256), dtype=np.uint8)
        y_test = np.empty((len(image_paths_test), 256, 256), dtype=np.uint8)
    else:
        X_train, X_test, y_train, y_test = list(), list(), list(), list()

    if load_into_ram:
        for i, image in enumerate(image_paths_train):
            img = tifffile.imread(image)
            if as_numpy:
                X_train[i] = img
            else:
                X_train.append(img)

        for i, image in enumerate(image_paths_test):
            img = tifffile.imread(image)
            if as_numpy:
                X_test[i] = img
            else:
                X_test.append(img)

        for i, label in enumerate(label_paths_train):
            label_img = tifffile.imread(label)
            if as_numpy:
                y_train[i] = label_img
            else:
                y_train.append(label_img)

        for i, label in enumerate(label_paths_test):
            label_img = tifffile.imread(label)
            if as_numpy:
                y_test[i] = label_img
            else:
                y_test.append(label_img)

        if as_numpy:
            image_paths_train = np.array(image_paths_train)
            image_paths_test = np.array(image_paths_test)
            label_paths_train = np.array(label_paths_train)
            label_paths_test = np.array(label_paths_test)

        # Convert to numpy arrays if requested
        if verbose:
            print("after loading images:", flush=True)
            print_available_memory()
            print(f"X_train size: {len(X_train)}")
            print(f"X_test size: {len(X_test)}")
            print(f"y_train size: {len(y_train)}")
            print(f"y_test size: {len(y_test)}")
            print(f"Image paths: {f'training image paths: {image_paths_train[:2]}'}")
            print(f"Label paths: {f'training label paths: {label_paths_train[:2]}'}")
            print(f"Image paths test: {f'test image paths: {image_paths_test[:2]}'}")
            print(
                f"Label paths test: {f'test label paths: {label_paths_test[:2]}'}",
                flush=True,
            )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            image_paths_train,
            image_paths_test,
            label_paths_train,
            label_paths_test,
        )
    else:
        if as_numpy:
            image_paths_train = np.array(image_paths_train)
            image_paths_test = np.array(image_paths_test)
            label_paths_train = np.array(label_paths_train)
            label_paths_test = np.array(label_paths_test)
        if verbose:
            print_available_memory()
            print(f"Image paths: {f'training image paths: {image_paths_train[:2]}'}")
            print(f"Label paths: {f'training label paths: {label_paths_train[:2]}'}")
            print(f"Image paths test: {f'test image paths: {image_paths_test[:2]}'}")
            print(
                f"Label paths test: {f'test label paths: {label_paths_test[:2]}'}",
                flush=True,
            )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            image_paths_train,
            image_paths_test,
            label_paths_train,
            label_paths_test,
        )


if __name__ == "__main__":
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR", ".")
    example_path = "/scratch/tmp/sluttman/Brick_Data_Train/"
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path,
    ) = load_split(example_path, load_into_ram=True)
