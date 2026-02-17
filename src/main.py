import cv2
import time
from pathlib import Path

# Project root = folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_ROOT / "data" / "Orings"
OUTPUT_FOLDER = PROJECT_ROOT / "output"


def process_image(image_path: Path):
    start_time = time.perf_counter()

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    output_img = img.copy()

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0
    return output_img, processing_time_ms


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for i in range(1, 16):
        filename = f"Oring{i}.jpg"
        image_path = DATA_FOLDER / filename

        result = process_image(image_path)
        if result is None:
            continue

        output_img, ms = result
        output_path = OUTPUT_FOLDER / f"output_{filename}"
        cv2.imwrite(str(output_path), output_img)

        print(f"{filename} processed in {ms:.2f} ms")


if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_FOLDER  =", DATA_FOLDER)
    main()
