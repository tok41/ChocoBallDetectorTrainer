import shutil
from pathlib import Path

import pytest
from tools import split_dataset

INPUT_DIR = "test/test_split_data/res_images"
TRAIN_DIR = "test/test_split_data/train"
TEST_DIR = "test/test_split_data/test"
IMG_EXT = "JPG"


@pytest.fixture
def clear_out_dir():
    out_path = Path(TRAIN_DIR)
    if out_path.exists():
        shutil.rmtree(TRAIN_DIR)
    out_path.mkdir(parents=True)

    out_path = Path(TEST_DIR)
    if out_path.exists():
        shutil.rmtree(TEST_DIR)
    out_path.mkdir(parents=True)


def test_split_dataset(clear_out_dir):
    path_train = Path(TRAIN_DIR)
    path_test = Path(TEST_DIR)
    split_dataset.split_dataset(INPUT_DIR, TRAIN_DIR, TEST_DIR)
    assert path_train.exists()
    assert path_test.exists()
    assert len(list(path_train.glob(f"*.{IMG_EXT}"))) == 3
    assert len(list(path_test.glob(f"*.{IMG_EXT}"))) == 1
    assert len(list(path_train.glob("*.xml"))) == 3
    assert len(list(path_test.glob("*.xml"))) == 1
