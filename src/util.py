"""Utilities for ChocoBallDetector
"""

import logging
from pathlib import Path


def set_logger(logger, log=None):
    """logger

    Args:
        log (str): (Optional) log file name,
            None->標準出力のみ, 値が入ればそのファイル名でログファイルを出力する
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger.setLevel(logging.DEBUG)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(handler1)

    if log is not None:
        handler2 = logging.FileHandler(filename=log, mode="w")
        handler2.setFormatter(logging.Formatter(log_fmt))
        logger.addHandler(handler2)


def check_file(file_path):
    """fileの存在確認

    Args:
        file_path (str): ファイルパス

    Raises:
        FileNotFoundError: file does not exist
    """
    f_path = Path(file_path)
    if not f_path.exists():
        print(f"file not found : {file_path}")
        raise FileNotFoundError(f"{file_path}")
    return f_path


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
