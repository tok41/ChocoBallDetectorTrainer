import shutil
from pathlib import Path

import pytest
from PIL import Image
from tools import image_resizer

INPUT_DIR = "test/test_resizer/org_images"
OUT_DIR = "test/test_resizer/res_images"
IMG_EXT = "JPG"
RES_SHAPE = (402, 302)


@pytest.fixture
def clear_out_dir():
    out_path = Path(OUT_DIR)
    if out_path.exists():
        shutil.rmtree(OUT_DIR)
    out_path.mkdir(parents=True)


def test_resize_images(clear_out_dir):
    actual = image_resizer.resize_images(
        input_dir=INPUT_DIR, out_dir=OUT_DIR, img_ext=IMG_EXT, shape=RES_SHAPE
    )
    assert len(actual) == 4
    for img_path in actual:
        img = Image.open(img_path)
        assert img.width == RES_SHAPE[0]
        assert img.height == RES_SHAPE[1]
