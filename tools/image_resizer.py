"""画像データの準備のためのリサイズ
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def resize_images(input_dir, out_dir, img_ext, shape):
    logger.info(f"input: {input_dir}")
    logger.info(f"output: {out_dir}")
    path_input = Path(input_dir)
    path_out = Path(out_dir)
    image_files = list(path_input.glob(f"*.{img_ext}"))
    logger.info(f"images: {len(image_files)}")
    res_images = []
    for img_path in image_files:
        img = Image.open(img_path)
        if img.height > img.width:  # 向きを一定にする
            img = img.rotate(90, expand=True)
        res_img = img.resize(shape)
        out_file_path = path_out / img_path.name
        res_img.save(out_file_path)
        res_images.append(out_file_path)
        logger.info(str(out_file_path))
    return res_images


def command_parse():
    import argparse

    parser = argparse.ArgumentParser(description="argument image resizer")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="test/test_resizer/org_images",
        help="input directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="test/test_resizer/res_images",
        help="input directory",
    )
    parser.add_argument(
        "--img_ext", type=str, default="JPG", help="image file extension"
    )
    parser.add_argument("--width", type=str, default=402, help="resized image width")
    parser.add_argument("--height", type=str, default=302, help="resized image width")
    args = parser.parse_args()

    return args


def main():
    args = command_parse()
    input_dir = args.input_dir
    out_dir = args.out_dir
    img_ext = args.img_ext
    width = args.width
    height = args.height

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.INFO)

    resize_images(
        input_dir=input_dir,
        out_dir=out_dir,
        img_ext=img_ext,
        shape=(width, height),
    )

    return 0


if __name__ == "__main__":
    main()
