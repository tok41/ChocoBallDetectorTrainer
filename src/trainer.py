"""ChocoBallDetectorのモデル学習をする
基本的にはchainercvの公式exampleのコピー
そこそこwarningが出ているが、chainer側の問題なので、ここでは対処しない。
- モデル定義
- Optimizerの定義
- 学習の実施
"""

import logging
from pathlib import Path

import chainer
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainercv import transforms
from chainercv.chainer_experimental.datasets.sliceable import TupleDataset
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain

from src import util


class Transform(object):
    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (o_H, o_W), x_flip=params["x_flip"])

        return img, bbox, label, scale


class ChocoTrainer:
    def __init__(
        self,
        log_interval=1,
        plot_interval=1,
        print_interval=1,
        snap_shot_interval=10,
        step_size=100,
        out="result",
        n_epoch=20,
        gpu=0,
        logger=None,
    ):
        """init

        Args:
            gpu (int, optional): GPU ID. if gpu<0 then use CPU. Defaults to 0.
            train_rate (float, optional): 全データにおける訓練データの割合. Defaults to 0.8.
            logger (logger, optional): logger. Defaults to None.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.gpu = gpu
        self.dataset = None
        self.optimizer = None
        self.n_epoch = n_epoch
        self.set_output_directory(out)
        self.step_size = step_size
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.print_interval = print_interval
        self.snap_shot_interval = snap_shot_interval

    def set_output_directory(self, out):
        _ = util.check_dir(out, mkdir=True)
        self.out = out
        self.logger.info(f"set output: {out}")

    def set_data(self, images, bboxs, obj_ids):
        """データセットを登録する

        Args:
            images (numpy.ndarray): 画像データ. [N, channel, height, width]
            bboxs (list(numpy.ndarray)): ボウンディングボックスのリスト
                        type = [np.array([N, 4])],
                        bbox_axis = [y_min, x_min, y_max, x_max]
            obj_ids (list(numpy.ndarray)): バウンディングボックス毎の物体ID
                        type = [np.array(N, 1)]

        Returns:
            [TupleDataset]: dataset
        """
        if images.shape[0] != len(bboxs):
            raise ValueError(
                f"input size does not match: images={images.shape[0]}, "
                f"bounding_boxs={len(bboxs)}"
            )
        if images.shape[0] != len(obj_ids):
            raise ValueError(
                f"input size does not match: images={images.shape[0]}, "
                f"object_ids={len(obj_ids)}"
            )
        dataset = TupleDataset(("img", images), ("bbox", bboxs), ("label", obj_ids))
        self.dataset = dataset
        self.logger.info(f"set_dataset: {len(dataset)}")
        self.logger.info(f"set_dataset(leys): {dataset.keys}")
        return dataset

    def set_model(self, n_class=2):
        """モデルのセット

        Args:
            n_class (int, optional): 認識する物体クラスの数. Defaults to 2.
        """
        faster_rcnn = FasterRCNNVGG16(n_fg_class=n_class, pretrained_model="imagenet")
        faster_rcnn.use_preset("evaluate")
        model = FasterRCNNTrainChain(faster_rcnn)
        self.model = model
        self.logger.info("set FasterRCNNVGG16, pretrained=imagenet")

    def set_optimizer(self, lr=0.001, momentum=0.9):
        """optimizer

        Args:
            lr (float, optional): learning rate. Defaults to 0.001.
            momentum (float, optional): momentum coefficient. Defaults to 0.9.
        """
        optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))
        self.optimizer = optimizer
        self.logger.info("set Optimizer: MomentumSGD")

    def run(self):
        """学習処理の実行"""
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
            self.logger.info(f"use GPU: {self.gpu}")
        # データ分割
        if self.dataset is None:
            raise ValueError("dataset is not initialize. ")
        N = len(self.dataset)
        self.logger.info(f"the num of train-dataset: {N}")
        # iteratorのセット
        train_data = TransformDataset(self.dataset, Transform(self.model.faster_rcnn))
        train_iter = chainer.iterators.SerialIterator(train_data, batch_size=1)
        self.logger.info("set iterator")
        # updater
        if self.optimizer is None:
            raise ValueError("optimizer is not initialize. ")
        updater = chainer.training.updaters.StandardUpdater(
            train_iter, self.optimizer, device=self.gpu
        )
        self.logger.info(f"optimizer: {type(self.optimizer)}")
        # trainer
        trainer = chainer.training.Trainer(
            updater, (self.n_epoch, "epoch"), out=self.out
        )
        self.logger.info(f"epochs: {self.n_epoch}")
        trainer.extend(
            extensions.snapshot_object(
                self.model.faster_rcnn, "snapshot_epoch_{.updater.epoch}.npz"
            ),
            trigger=(self.snap_shot_interval, "epoch"),
        )  # 学習途中のスナップショット
        trainer.extend(
            extensions.ExponentialShift("lr", 0.1),
            trigger=(self.step_size, "iteration"),
        )  # lrを低減させる
        trainer.extend(
            extensions.observe_lr(), trigger=(self.log_interval, "epoch")
        )  # lrのロギング
        trainer.extend(
            extensions.LogReport(trigger=(self.log_interval, "epoch"))
        )  # trainerの経過状況をファイルに書き出す
        trainer.extend(
            extensions.PrintReport(
                [
                    "iteration",
                    "epoch",
                    "elapsed_time",
                    "lr",
                    "main/loss",
                    "main/roi_loc_loss",
                    "main/roi_cls_loss",
                    "main/rpn_loc_loss",
                    "main/rpn_cls_loss",
                    "validation/main/map",
                ]
            ),
            trigger=(self.print_interval, "epoch"),
        )
        trainer.extend(
            extensions.PlotReport(
                ["main/loss"],
                file_name="loss.png",
                trigger=(self.plot_interval, "epoch"),
            ),
            trigger=(self.plot_interval, "epoch"),
        )  # lossのplot
        trainer.extend(extensions.dump_graph("main/loss"))
        # training
        self.logger.info("run training")
        trainer.run()

    def save_model(self, file_name):
        """モデルをファイル出力する

        Args:
            file_name (str): file path
        """
        path_file = Path(file_name)
        if path_file.exists():
            self.logger.info(f"overwrite(file exist): {file_name}")
        chainer.serializers.save_npz(str(path_file), self.model.faster_rcnn)
        self.logger.info(f"output model as npz_file: {file_name}")
