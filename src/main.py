import cv2
import time
from pathlib import Path
import numpy as np

from histogram import compute_histogram, otsu_threshold


# Project root = folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_ROOT / "data" / "Orings"
OUTPUT_FOLDER = PROJECT_ROOT / "output"


def process_image(image_path: Path):
    start_time = time.perf_counter()

    # Load image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    # --- Step 1: Compute histogram (NumPy only) ---
    hist = compute_histogram(img)

    # --- Step 2: Automatic threshold (Otsu) ---
    threshold = otsu_threshold(hist)

    # --- Step 3: Apply threshold ---
    # Assuming O-ring is darker than background
    binary = (img < threshold).astype(np.uint8) * 255

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0

    return binary, processing_time_ms, threshold


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_FOLDER  =", DATA_FOLDER)

    for i in range(1, 16):
        filename = f"Oring{i}.jpg"
        image_path = DATA_FOLDER / filename

        result = process_image(image_path)
        if result is None:
            continue

        binary_img, ms, threshold = result

        output_path = OUTPUT_FOLDER / f"binary_{filename}"
        cv2.imwrite(str(output_path), binary_img)

        print(f"{filename} | threshold={threshold} | {ms:.2f} ms")


if __name__ == "__main__":
    main()
