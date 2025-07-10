import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm

def create_patches(image_dir, mask_dir, output_image_dir, output_mask_dir,
                   patch_size=64, stride=32, max_patches_per_image=None,
                   start_idx=0, end_idx=None, use_offset=True, filter_empty=True):
    """
    Erstellt Patches aus Sentinel-2 Bildern und den zugehörigen Masken.
    Mit versetztem Stride für mehr Diversität und optionalem Filter für leere Patches.

    Args:
        image_dir (str): Pfad zu den Originalbildern.
        mask_dir (str): Pfad zu den Masken.
        output_image_dir (str): Zielordner für Bild-Patches.
        output_mask_dir (str): Zielordner für Masken-Patches.
        patch_size (int): Größe der quadratischen Patches.
        stride (int): Schrittweite beim Sliding Window.
        max_patches_per_image (int): Max. Anzahl Patches pro Bild (Optional).
        start_idx (int): Startindex der Bilder (für Parallelisierung).
        end_idx (int): Endindex der Bilder (für Parallelisierung).
        use_offset (bool): Aktiviert versetzten Stride.
        filter_empty (bool): Speichert nur Patches mit mind. 1 Nicht-Background-Pixel.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

    if end_idx is None or end_idx > len(image_files):
        end_idx = len(image_files)

    selected_images = image_files[start_idx:end_idx]
    selected_masks = mask_files[start_idx:end_idx]

    print(f"📂 Bearbeite {len(selected_images)} Bilder ({start_idx} bis {end_idx})")

    for img_name, mask_name in tqdm(zip(selected_images, selected_masks), total=len(selected_images)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image = tiff.imread(img_path)
        mask = tiff.imread(mask_path)

        H, W = mask.shape[:2]
        patches_saved = 0

        for y in range(0, H - patch_size + 1, stride):
            # Offset jede zweite Zeile
            x_offset = stride if use_offset and ((y // stride) % 2 == 1) else 0

            for x in range(x_offset, W - patch_size + 1, stride):
                img_patch = image[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                # Filter: nur Patches mit Nicht-Background-Pixeln
                if filter_empty and np.all(mask_patch == 0):
                    continue

                patch_id = f"{os.path.splitext(img_name)[0]}_{y}_{x}"
                img_out_path = os.path.join(output_image_dir, f"{patch_id}.tif")
                mask_out_path = os.path.join(output_mask_dir, f"{patch_id}.tif")

                tiff.imwrite(img_out_path, img_patch)
                tiff.imwrite(mask_out_path, mask_patch)

                patches_saved += 1
                if max_patches_per_image and patches_saved >= max_patches_per_image:
                    break
            if max_patches_per_image and patches_saved >= max_patches_per_image:
                break

        print(f"✅ {img_name}: {patches_saved} Patches gespeichert")

if __name__ == "__main__":
    DATA_DIR = "/scratch/tmp/sluttman/Brick_Data_Train"
    PATCH_DIR = "/scratch/tmp/sluttman/Brick_Patches_Filtered"

    IMAGE_DIR = os.path.join(DATA_DIR, "Image")
    MASK_DIR = os.path.join(DATA_DIR, "Mask")
    PATCH_IMAGE_DIR = os.path.join(PATCH_DIR, "Image")
    PATCH_MASK_DIR = os.path.join(PATCH_DIR, "Mask")

    create_patches(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        output_image_dir=PATCH_IMAGE_DIR,
        output_mask_dir=PATCH_MASK_DIR,
        patch_size=64,
        stride=32,
        max_patches_per_image=500,
        start_idx=0,
        end_idx=None,
        use_offset=True,    # 🔥 versetzten Stride aktivieren
        filter_empty=True   # 🔥 leere Patches aussortieren
    )
