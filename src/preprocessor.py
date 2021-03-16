"""ChocoballDetectorの前処理モジュール
"""

import logging
from collections import OrderedDict

import numpy as np
import xmltodict
from PIL import Image
from tqdm import tqdm

from src import util


class ChocoPreProcessor:
    def __init__(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.classes = None

    def set_classes(self, class_file):
        """認識物体のカテゴリ名をセットする

        Args:
            class_file (str or pathlib): 物体クラスの定義ファイル(テキスト).
                    1行毎に物体クラスを文字列で入力.

        Returns:
            dict{物体クラス名:クラスID}: 物体クラス
        """
        self.logger.info(f"set object class: {class_file}")
        _ = util.check_file(class_file)
        classes = dict()
        with open(class_file) as fd:
            for i, one_line in enumerate(fd.readlines()):
                cl = one_line.split("\n")[0]
                classes[cl] = i
        self.classes = classes
        self.logger.info(f"classes: {classes.keys()}")
        return classes

    def get_annotation_files(self, dir, ext="xml"):
        """アノテーションデータファイルのリストをPathlib.Path形式で取得する

        Args:
            dir (str): アノテーションデータの格納されているディレクトリのパス
            ext (str, optional): アノテーションデータファイルの拡張子.
                    Defaults to "xml".

        Raises:
            FileNotFoundError: アノテーションデータファイルが見つからなかった

        Returns:
            list(pathlib.Path): アノテーションデータファイルのリスト
        """
        path_anno = util.check_dir(dir)
        files = list(path_anno.glob(f"**/*.{ext}"))
        if len(files) < 1:
            raise FileNotFoundError("Annotation File does not exists.")
        return files

    def get_bounding_box_data(self, xml_file):
        """アノテーションデータファイルから情報をパースする

        Args:
            xml_file (str or pathlib.Path): アノテーションファイルのパス

        Returns:
            numpy.array: バウンディングボックスの座標.
                    shape=(N,4), [N, [y_min, x_min, y_max, x_max]]
        """
        _ = util.check_file(xml_file)
        with open(xml_file) as f:
            tmp = xmltodict.parse(f.read())
        annotation = tmp["annotation"]
        bboxs = np.array(
            [
                [
                    obj["bndbox"]["ymin"],
                    obj["bndbox"]["xmin"],
                    obj["bndbox"]["ymax"],
                    obj["bndbox"]["xmax"],
                ]
                for obj in annotation["object"]
            ],
            dtype=np.float32,
        )
        obj_names = np.array([obj["name"] for obj in annotation["object"]])
        meta_info = {
            "folder": annotation["folder"],
            "filename": annotation["filename"],
            "path": annotation["path"],
        }
        return bboxs, obj_names, meta_info

    def read_image(self, img_file):
        """画像データを読みchainerで扱えるndarrayに変換する

        Args:
            img_file (str or pathlib): 画像ファイルのパス

        Returns:
            numpy.ndarray: 画像データ.
                    shape=[channel, height, width]
                    dtype=float32
        """
        img_path = util.check_file(img_file)
        img = Image.open(img_path)
        img_arr = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img_arr

    def get_object_ids(self, obj_names):
        """オブジェクトIDを取得する

        Args:
            obj_names (list(str)): 物体カテゴリ名のリスト

        Raises:
            ValueError: 物体カテゴリが定義されていない場合

        Returns:
            numpy.array: オブジェクトIDのarray
        """
        if self.classes is None:
            raise ValueError(
                "Classes is None. You should set classes (use set_classes). "
            )
        ids = np.array(
            [self.classes[obj_name] for obj_name in obj_names], dtype=np.int32
        )
        return ids

    def set_dataset(self, anno_dir, img_dir, img_ext="JPG"):
        """学習用のデータセットを用意する

        Args:
            anno_dir (str or pathlib): アノテーションデータのディレクトリ
            img_dir (str or pathlib): 画像ディレクトリ
            img_ext (str, optional): 画像ファイルの拡張子. Defaults to "JPG".

        Raises:
            ValueError: class定義がされていない場合

        Returns:
            OrderedDict: 学習データセット
                        keys: 画像ID. 画像ファイル名の拡張子なしの文字列.
                        value: dict{keys=["image", "bboxs", "obj_names", "obj_ids"]}
        """
        if self.classes is None:
            raise ValueError(
                "Classes is None. You should set classes (use set_classes). "
            )
        path_anno = util.check_dir(anno_dir)
        path_img = util.check_dir(img_dir)
        self.path_anno = path_anno
        self.path_img = path_img
        self.logger.info(f"annotation_file_path: {anno_dir}")
        self.logger.info(f"image_file_path: {img_dir}")

        anno_files = self.get_annotation_files(anno_dir)
        self.annotation_files = anno_files
        self.logger.info(f"annotation_file_size: {len(anno_files)}")

        dataset = OrderedDict()
        for anno_file in tqdm(anno_files):
            bboxs, obj_names, meta_info = self.get_bounding_box_data(xml_file=anno_file)
            obj_ids = self.get_object_ids(obj_names=obj_names)
            img_file = path_img / meta_info["filename"]
            try:
                img_arr = self.read_image(img_file=img_file)
            except FileNotFoundError:
                self.logger.warning(
                    f"Image file does not exist for inputted xml_file. : {str(img_file)}"
                )
                continue
            image_id = img_file.stem
            dataset[image_id] = {
                "image": img_arr,
                "bboxs": bboxs,
                "obj_names": obj_names,
                "obj_ids": obj_ids,
            }
        self.dataset = dataset
        return dataset

    def get_bbox_list(self):
        """バウンディングボックス座標のarrayをリストにして返す

        Returns:
            list(np.array): バウンディングボックス座標のリスト
        """
        bbox_array = [val["bboxs"] for k, val in self.dataset.items()]
        return bbox_array

    def get_img_array(self):
        """画像データをarrayにして返す

        Returns:
            numpy.array: 画像データ. [N, channel, height, width]
        """
        img_array = np.array([val["image"] for k, val in self.dataset.items()])
        return img_array

    def get_object_ids_list(self):
        """画像毎の物体IDをリストにして返す

        Returns:
            list(np.array): 物体IDのarrayのリスト
        """
        obj_ids = [val["obj_ids"] for k, val in self.dataset.items()]
        return obj_ids

    def get_object_classes(self):
        """物体クラス名のリストを返す

        Returns:
            list(str): 物体クラスのリスト
        """
        classes_list = list(self.classes.keys())
        return classes_list
