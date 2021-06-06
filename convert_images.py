import os
import pathlib
import json
import zipfile

from tqdm import tqdm
import cv2
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from config_data import ConvertConfig
from utils import load_json

cs = ConfigStore()
cs.store("train_detector", node=ConvertConfig)


def decode_image(buffer):
    buffer = np.frombuffer(buffer, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


@hydra.main(config_name="convert_images")
def main(convert_config: ConvertConfig):
    os.chdir(hydra.utils.get_original_cwd())

    out_dir = pathlib.Path(convert_config.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with zipfile.ZipFile(convert_config.input_zip, "r") as zip_file:
        with zip_file.open("data/train.json", "r") as plate_info_file:
            plates_info = json.load(plate_info_file)

        remove_plates = []
        total_images = 0
        for i, plate_info in tqdm(enumerate(plates_info), total=len(plates_info)):
            path = f"data/{plate_info['file']}"
            with zip_file.open(path, "r") as img_file:
                try:
                    img = decode_image(img_file.read())
                except Exception:
                    img = None

                if img is None:
                    print("Detected incorect image", plate_info['file'])
                    remove_plates.append(i)
                else:
                    new_file = pathlib.Path(path).relative_to("data")
                    img_path = out_dir / new_file
                    img_path.parent.mkdir(exist_ok=True, parents=True)
                    img_path = img_path.with_suffix(".jpg")
                    plate_info["file"] = str(img_path.relative_to(out_dir).as_posix())
                    cv2.imwrite(str(img_path), img, (cv2.IMWRITE_JPEG_OPTIMIZE, 1))
                    total_images += 1

        print("Convert train images: ", total_images)

        plates_info = [plate for i, plate in enumerate(
            plates_info) if i not in remove_plates]

        with open(out_dir / "train.json", "w") as file:
            json.dump(plates_info, file)

        path = "data/test"

        total_images = 0

        for file in tqdm(zip_file.infolist()):
            if not file.is_dir() and file.filename.startswith(path):
                with zip_file.open(file.filename, "r") as img_file:
                    try:
                        img = decode_image(img_file.read())
                    except Exception:
                        img = None

                    if img is None:
                        print("Detected incorect image", file.filename)
                    else:
                        new_file = pathlib.Path(file.filename).relative_to("data")
                        img_path = out_dir / new_file
                        img_path.parent.mkdir(exist_ok=True, parents=True)
                        img_path = img_path.with_suffix(".jpg")
                        cv2.imwrite(str(img_path), img, (cv2.IMWRITE_JPEG_OPTIMIZE, 1))
                        total_images += 1

        print("Convert test images: ", total_images)


if __name__ == "__main__":
    main()
