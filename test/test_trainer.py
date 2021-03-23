import chainer
import pytest
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from src.preprocessor import ChocoPreProcessor
from src.trainer import ChocoTrainer


@pytest.fixture
def sample_trainer():
    tr = ChocoTrainer()
    return tr


@pytest.fixture
def sample_preprocess():
    TEST_IMG_DIR = "test/test_preprocessor/imgs"
    TEST_BBOX_DIR = "test/test_preprocessor/bboxs"
    CLASSES_FILE = "data/classes.txt"
    cp = ChocoPreProcessor()
    _ = cp.set_classes(class_file=CLASSES_FILE)
    _ = cp.set_dataset(anno_dir=TEST_BBOX_DIR, img_dir=TEST_IMG_DIR)
    return cp


class TestSetDataset:
    def test_set_train_data(self, sample_trainer):
        sample_trainer.set_train_rate(0.5)
        assert sample_trainer.train_rate == 0.5

    def test_set_train_data_invalid_value(self, sample_trainer):
        with pytest.raises(ValueError):
            sample_trainer.set_train_rate(-0.1)
        with pytest.raises(ValueError):
            sample_trainer.set_train_rate(1.5)

    def test_set_data(self, sample_trainer, sample_preprocess):
        bboxs = sample_preprocess.get_bbox_list()
        imgs = sample_preprocess.get_img_array()
        obj_ids = sample_preprocess.get_object_ids_list()
        dataset = sample_trainer.set_data(imgs, bboxs, obj_ids)
        assert len(dataset) == 2
        assert dataset.keys == ("img", "bbox", "label")

    def test_set_data_invalid_size(self, sample_trainer, sample_preprocess):
        bboxs = sample_preprocess.get_bbox_list()
        imgs = sample_preprocess.get_img_array()
        obj_ids = sample_preprocess.get_object_ids_list()
        with pytest.raises(ValueError):
            sample_trainer.set_data(imgs, bboxs[:1], obj_ids)
        with pytest.raises(ValueError):
            sample_trainer.set_data(imgs, bboxs, obj_ids[:1])


class TestSetTrainSetting:
    def test_set_model(self, sample_trainer):
        sample_trainer.set_model()
        assert isinstance(sample_trainer.model, FasterRCNNTrainChain)

    def test_set_optimizer(self, sample_trainer):
        sample_trainer.set_model()
        sample_trainer.set_optimizer()
        assert isinstance(sample_trainer.optimizer, chainer.optimizers.MomentumSGD)
