from pathlib import Path

import numpy as np
import pytest
from src.preprocessor import ChocoPreProcessor

TEST_IMG_DIR = "test/test_preprocessor/imgs"
TEST_BBOX_DIR = "test/test_preprocessor/bboxs"
CLASSES_FILE = "data/classes.txt"


@pytest.fixture
def sample_choco_prep():
    cp = ChocoPreProcessor()
    _ = cp.set_classes(class_file=CLASSES_FILE)
    _ = cp.set_dataset(anno_dir=TEST_BBOX_DIR, img_dir=TEST_IMG_DIR)
    return cp


class TestSetClass:
    def test_set_classes(self):
        cp = ChocoPreProcessor()
        actual = cp.set_classes(class_file=CLASSES_FILE)
        assert isinstance(actual, dict)
        assert len(set(actual.keys()) - set(["choco-ball", "choco-package"])) < 1

    def test_set_classes_invalid_file_path(self):
        cp = ChocoPreProcessor()
        with pytest.raises(FileNotFoundError):
            _ = cp.set_classes(class_file="test/not_exist_file.txt")


class TestGetAnnotationFiles:
    def test_get_annotation_files(self):
        cp = ChocoPreProcessor()
        actual = cp.get_annotation_files(dir=TEST_BBOX_DIR)
        assert len(actual) == 2
        for p in actual:
            assert isinstance(p, Path)

    def test_get_annotation_files_not_exist_files(self):
        """アノテーションファイルが存在しないとき"""
        cp = ChocoPreProcessor()
        # ディレクトリが違う
        with pytest.raises(FileNotFoundError):
            _ = cp.get_annotation_files(dir=TEST_IMG_DIR)
        # 拡張子が違う
        with pytest.raises(FileNotFoundError):
            _ = cp.get_annotation_files(dir=TEST_BBOX_DIR, ext="bad_ext")

    def test_get_annotation_files_invalid_dir(self):
        """指定したディレクトリが存在しない"""
        cp = ChocoPreProcessor()
        with pytest.raises(FileNotFoundError):
            _ = cp.get_annotation_files(dir="test/not_exist_dir")


class TestGetBoundingBoxData:
    def test_get_bounding_box_data(self):
        cp = ChocoPreProcessor()
        file_paths = cp.get_annotation_files(dir=TEST_BBOX_DIR)
        bboxs, obj_names, meta_info = cp.get_bounding_box_data(
            xml_file=str(file_paths[0])
        )
        assert bboxs.shape[0] == obj_names.shape[0]
        assert bboxs.shape[1] == 4

    def test_get_bounding_box_data_input_type_pathlib(self):
        cp = ChocoPreProcessor()
        file_paths = cp.get_annotation_files(dir=TEST_BBOX_DIR)
        bboxs, obj_names, meta_info = cp.get_bounding_box_data(xml_file=file_paths[0])
        assert bboxs.shape[0] == obj_names.shape[0]
        assert bboxs.shape[1] == 4

    def test_get_bounding_box_data_invalid_filename(self):
        cp = ChocoPreProcessor()
        with pytest.raises(FileNotFoundError):
            _ = cp.get_bounding_box_data(xml_file="test/not_exist_file.xml")


class TestReadImage:
    def test_read_image(self):
        path_img = Path(TEST_IMG_DIR)
        img_path = path_img / "IMG_2787.JPG"
        cp = ChocoPreProcessor()
        actual = cp.read_image(img_path)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (3, 302, 402)

    def test_read_image_str_input(self):
        path_img = Path(TEST_IMG_DIR)
        img_path = path_img / "IMG_2787.JPG"
        cp = ChocoPreProcessor()
        actual = cp.read_image(str(img_path))
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (3, 302, 402)


class TestGetObjIDs:
    def test_get_object_ids(self):
        cp = ChocoPreProcessor()
        _ = cp.set_classes(class_file=CLASSES_FILE)
        dummy_names = ["choco-ball", "choco-ball", "choco-package"]
        expect = np.array([0, 0, 1], dtype=np.int32)
        actual = cp.get_object_ids(dummy_names)
        assert (actual == expect).all()


class TestSetDataset:
    def test_set_dataset(self):
        cp = ChocoPreProcessor()
        _ = cp.set_classes(class_file=CLASSES_FILE)
        actual = cp.set_dataset(anno_dir=TEST_BBOX_DIR, img_dir=TEST_IMG_DIR)
        assert list(actual.keys()) == ["IMG_2914", "IMG_2915"]
        for key, val in actual.items():
            assert list(val.keys()) == ["image", "bboxs", "obj_names", "obj_ids"]
            assert val["image"].shape == (3, 302, 402)
            assert val["bboxs"].shape[1] == 4

    def test_set_dataset_much_xml_file(self):
        cp = ChocoPreProcessor()
        _ = cp.set_classes(class_file=CLASSES_FILE)
        much_bbox_dir = "test/test_preprocessor/bboxs_much"
        actual = cp.set_dataset(anno_dir=much_bbox_dir, img_dir=TEST_IMG_DIR)
        assert list(actual.keys()) == ["IMG_2914", "IMG_2915"]


class TestArrayData:
    def test_get_bbox_array(self, sample_choco_prep):
        actual = sample_choco_prep.get_bbox_list()
        assert len(actual) == 2
        assert actual[0].shape == (20, 4)
        assert actual[1].shape == (19, 4)

    def test_get_img_array(self, sample_choco_prep):
        actual = sample_choco_prep.get_img_array()
        assert actual.shape == (2, 3, 302, 402)

    def test_get_object_ids_array(self, sample_choco_prep):
        actual = sample_choco_prep.get_object_ids_list()
        assert len(actual) == 2
        assert len(actual[0]) == 20
        assert len(actual[1]) == 19

    def test_get_object_classes(self, sample_choco_prep):
        actual = sample_choco_prep.get_object_classes()
        assert isinstance(actual, list)
        assert actual == ["choco-ball", "choco-package"]
