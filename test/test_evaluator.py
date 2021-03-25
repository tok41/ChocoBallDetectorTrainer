import chainercv
import numpy as np
import pytest
from src.evaluator import ChocoEvaluator
from src.preprocessor import ChocoPreProcessor

SAMPLE_MODEL = "test/assets/choco_faster_rcnn.npz"
SAMPLE_DATA = "test/assets/res_images"
CLASSES_FILE = "data/classes.txt"


@pytest.fixture(scope="module")
def sample_evaluator():
    ce = ChocoEvaluator()
    ce.load_model(model_file=SAMPLE_MODEL)
    return ce


@pytest.fixture
def sample_dataset():
    cp = ChocoPreProcessor()
    _ = cp.set_classes(class_file=CLASSES_FILE)
    _ = cp.set_dataset(anno_dir=SAMPLE_DATA, img_dir=SAMPLE_DATA)
    return cp


class TestLoadModel:
    def test_load_model(self, sample_evaluator):
        assert isinstance(
            sample_evaluator.model,
            chainercv.links.model.faster_rcnn.faster_rcnn_vgg.FasterRCNNVGG16,
        )


class TestInference:
    def test_detect_choco_ball(self, sample_evaluator, sample_dataset):
        imgs = sample_dataset.get_img_array()
        actual = sample_evaluator.detect_choco_ball(img=imgs[0])
        assert isinstance(actual, dict)
        expect_keys = ["image", "pred_bboxs", "pred_labels", "scores"]
        assert len(set(actual.keys() - set(expect_keys))) < 1

    def test_detect_choco_balls(self, sample_evaluator, sample_dataset):
        imgs = sample_dataset.get_img_array()
        actual = sample_evaluator.detect_choco_balls(images=imgs[:2])
        assert isinstance(actual, list)
        assert len(actual) == 2

    def test_diff_choco_num(self):
        ce = ChocoEvaluator()
        dummy_label_pred = np.array([0, 0, 0, 1])
        dummy_label_true = np.array([1, 0, 0, 0])
        actual = ce.diff_chocoball_number(
            pred_labels=dummy_label_pred,
            true_labels=dummy_label_true,
            target_label_id=0,
        )
        assert actual == 0

        dummy_label_true = np.array([1, 0, 0, 0, 0])
        actual = ce.diff_chocoball_number(
            pred_labels=dummy_label_pred,
            true_labels=dummy_label_true,
            target_label_id=0,
        )
        assert actual == -1

        dummy_label_true = np.array([0, 0, 0])
        actual = ce.diff_chocoball_number(
            pred_labels=dummy_label_pred,
            true_labels=dummy_label_true,
            target_label_id=0,
        )
        assert actual == 0

    def test_diff_choco_num_input_type_invalid(self):
        ce = ChocoEvaluator()
        dummy_label_pred = [0, 0, 0, 1]
        dummy_label_true = np.array([1, 0, 0, 0])
        with pytest.raises(TypeError):
            _ = ce.diff_chocoball_number(
                pred_labels=dummy_label_pred,
                true_labels=dummy_label_true,
                target_label_id=0,
            )
        dummy_label_pred = np.array([0, 0, 0, 1])
        dummy_label_true = [1, 0, 0, 0]
        with pytest.raises(TypeError):
            _ = ce.diff_chocoball_number(
                pred_labels=dummy_label_pred,
                true_labels=dummy_label_true,
                target_label_id=0,
            )


class TestEvaluation:
    def test_mean_square_error(self):
        pred = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
        ]
        true = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
        ]
        ce = ChocoEvaluator()
        actual = ce.mean_square_error(pred_labels=pred, true_labels=true)
        assert actual == 0.0
        actual = ce.mean_square_error(
            pred_labels=pred, true_labels=true, target_label_id=1
        )
        assert actual == 0.5

    def test_mean_square_error2(self):
        pred = [
            np.array([0, 0, 0, 1]),
            np.array([0, 1]),
        ]
        true = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
        ]
        ce = ChocoEvaluator()
        actual = ce.mean_square_error(pred_labels=pred, true_labels=true)
        assert actual == 2.0

    def test_mean_square_error3(self):
        pred = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 0, 1]),
        ]
        true = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
        ]
        ce = ChocoEvaluator()
        actual = ce.mean_square_error(pred_labels=pred, true_labels=true)
        assert actual == 0.5
