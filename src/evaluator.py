"""ChocoBallDetectorのモデルを評価する
"""

import logging

import chainer
import matplotlib.pyplot as plt
import numpy as np
from chainercv.links import FasterRCNNVGG16
from chainercv.visualizations import vis_bbox

from src import util


class ChocoEvaluator:
    def __init__(
        self,
        gpu=-1,
        logger=None,
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.gpu = gpu

    def load_model(self, model_file, n_class=2):
        """Detectorモデルをロードする

        Args:
            model_file (str): モデルファイルパス
            n_class (int, optional): 認識物体の数. Defaults to 2.
        """
        _ = util.check_file(model_file)
        model = FasterRCNNVGG16(n_fg_class=n_class, pretrained_model=model_file)
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            model.to_gpu()
            self.logger.info(f"use GPU: {self.gpu}")
        self.model = model
        self.model_file = model_file

    def detect_choco_ball(self, img):
        """imageデータを入力して検出結果を辞書形式で返す

        Args:
            img (numpy.ndarray): イメージデータ. [channel, height, width]

        Returns:
            dict: keys=["image": np.ndarray,
                        "pred_bboxs": np.ndarray(N, axis),
                        "pred_labels": np.ndarray(N, ),
                        "scores": np.ndarray(N, ), ]
        """
        prd_bboxes, prd_labels, prd_scores = self.model.predict([img])
        return {
            "image": img,
            "pred_bboxs": prd_bboxes[0],
            "pred_labels": prd_labels[0],
            "scores": prd_scores[0],
        }

    def detect_choco_balls(self, images):
        """複数のimageデータを入力して検出結果を返す

        Args:
            images (numpy.ndarray): イメージデータ. [N, channel, height, width]

        Returns:
            list(dict): 各イメージデータの検出結果
        """
        prd_bboxes, prd_labels, prd_scores = self.model.predict(images)
        N = images.shape[0]
        lst_res = []
        for i in range(N):
            res = {
                "image": images[i],
                "pred_bboxs": prd_bboxes[i],
                "pred_labels": prd_labels[i],
                "scores": prd_scores[i],
            }
            lst_res.append(res)
        return lst_res

    def diff_chocoball_number(self, pred_labels, true_labels, target_label_id=0):
        """チョコボール検出個数の誤差を算出

        Args:
            pred_labels (numpy.array): 予測ラベル
            true_labels (numpy.array): 正解ラベル
            target_label_id (int, optional): 対象ラベルID. Defaults to 0.

        Raises:
            TypeError: pred_labelの型がnp.arrayでないとき
            TypeError: true_labelの型がnp.arrayでないとき

        Returns:
            int: 検出個数の誤差
        """
        if not isinstance(pred_labels, np.ndarray):
            raise TypeError(
                f"type of pred_label needs numpy.array: {type(pred_labels)}"
            )
        if not isinstance(true_labels, np.ndarray):
            raise TypeError(
                f"type of true_label needs numpy.array: {type(true_labels)}"
            )
        pred_num = np.sum(pred_labels == target_label_id)
        true_num = np.sum(true_labels == target_label_id)
        diff = pred_num - true_num
        return diff

    def mean_square_error(self, pred_labels, true_labels, target_label_id=0):
        """MSEを算出する

        Args:
            pred_labels (list(np.array)): 予測ラベルのarrayのlist
            true_labels (list(np.array)): 正解ラベルのarrayのlist
            target_label_id (int, optional): ターゲットラベル. Defaults to 0.

        Returns:
            float: MSE
        """
        diffs = [
            self.diff_chocoball_number(pred, true, target_label_id)
            for pred, true in zip(pred_labels, true_labels)
        ]
        mse = np.dot(diffs, diffs) / float(len(diffs))
        return mse

    def evaluate_chocoball_number(self, images, true_labels, target_label_id=0):
        """イメージarrayを入力して、評価結果を返す

        Args:
            images (numpy.array: ndarray of images. [N, channel, height, width]
            true_labels (list(numpy.array)): 正解ラベルのリスト
            target_label_id (int, optional): 計測対象のID. Defaults to 0.

        Returns:
            list(dict): 検出結果(辞書)のリスト
            float: MSE
        """
        result_list = self.detect_choco_balls(images=images)
        pred_labels = [res["pred_labels"] for res in result_list]
        mse = self.mean_square_error(
            pred_labels=pred_labels,
            true_labels=true_labels,
            target_label_id=target_label_id,
        )
        return result_list, mse

    def vis_detect_image(
        self, res, vis_score=True, classes=["choco-ball", "choco-package"], fig=None
    ):
        """検出結果の可視化

        Args:
            res (dict): 検出結果. keys=['image', 'pred_bboxs', 'pred_labels', 'scores']
            vis_score (bool, optional): スコアを表示するか否か. Defaults to True.
            classes (list, optional): 物体クラス名の定義.
                        Defaults to ["choco-ball", "choco-package"].
            fig (plt.Figure, optional): figureオブジェクト. Defaults to None.

        Returns:
            plt.Figure: figure
        """
        n_col = 1
        if vis_score:
            n_col = 2
        if fig is None:
            fig = plt.figure(figsize=(6 * n_col, 4))
        ax = fig.add_subplot(1, n_col, 1)
        vis_bbox(
            res["image"],
            res["pred_bboxs"],
            res["pred_labels"],
            ax=ax,
        )
        if vis_score:
            ax = fig.add_subplot(1, n_col, 2)
            vis_bbox(
                res["image"],
                res["pred_bboxs"],
                res["pred_labels"],
                res["scores"],
                label_names=classes,
                ax=ax,
            )
        return fig
