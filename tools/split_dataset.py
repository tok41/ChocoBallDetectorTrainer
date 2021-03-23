"""画像データを訓練用途評価ように分割する
"""

import logging
import random
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def check_dir(dir, mkdir=False):
    """check directory
    ディレクトリの存在を確認する。

    Args:
        dir (str): 対象ディレクトリのパス(文字列)
        mkdir (bool, optional): 存在しない場合にディレクトリを作成するかを指定するフラグ.
                                Defaults to False.

    Raises:
        FileNotFoundError: mkdir=Falseの場合に、ディレクトリが存在しない場合に例外をraise

    Returns:
        dir_path : ディレクトリのPathオブジェクト
    """
    dir_path = Path(dir)
    if not dir_path.exists():
        print(f"directory not found : {dir}")
        if mkdir:
            print(f"make directory : {dir}")
            dir_path.mkdir(parents=True)
        else:
            raise FileNotFoundError(f"{dir}")
    return dir_path


def _copy_file(file_paths, to_dir, img_ext="JPG", logger=logger):
    for fp in file_paths:
        shutil.copyfile(str(fp), str(to_dir / fp.name))
        logger.info(f"{str(fp)} to {str(to_dir)}")
        path_dir = fp.parent
        img_file_path = path_dir / f"{fp.stem}.{img_ext}"
        if img_file_path.exists():
            shutil.copyfile(str(img_file_path), str(to_dir / img_file_path.name))
        else:
            logger.warn(f"img file does not exist. : {str(img_file_path)}")


def split_dataset(
    image_dir, train_dir, test_dir, img_ext="JPG", train_rate=0.8, logger=logger
):
    anno_ext = "xml"
    path_img_d = check_dir(image_dir)
    path_train = check_dir(train_dir, mkdir=True)
    path_test = check_dir(test_dir, mkdir=True)
    annotation_files = np.array(list(path_img_d.glob(f"*.{anno_ext}")))

    N_annotation = len(annotation_files)
    N_train = (int)(N_annotation * train_rate)
    logger.info(f"the num of annotation data: {N_annotation}")
    logger.info(f"train rate: {train_rate}")
    logger.info(f"the num of train data: {N_train}")
    idxs = list(np.arange(N_annotation))
    random.shuffle(idxs)
    train_idxs = idxs[:N_train]
    test_idxs = idxs[N_train:]
    _copy_file(annotation_files[train_idxs], to_dir=path_train, logger=logger)
    _copy_file(annotation_files[test_idxs], to_dir=path_test, logger=logger)


def command_parse():
    import argparse

    parser = argparse.ArgumentParser(description="argument split_dataset")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="test/test_split_data/res_images",
        help="input directory",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="test/test_split_data/train",
        help="train directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="test/test_split_data/test",
        help="test directory",
    )
    parser.add_argument(
        "--img_ext", type=str, default="JPG", help="image file extension"
    )
    parser.add_argument(
        "--train_rate", type=float, default=0.8, help="resized image width"
    )
    args = parser.parse_args()

    return args


def main():
    args = command_parse()
    image_dir = args.image_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    img_ext = args.img_ext
    train_rate = args.train_rate

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.INFO)

    split_dataset(
        image_dir=image_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        img_ext=img_ext,
        train_rate=train_rate,
    )

    return 0


if __name__ == "__main__":
    main()
