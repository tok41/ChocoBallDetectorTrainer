import shutil
from pathlib import Path

import pytest
from src import util

TMP_DIR = "test/test_tmp_dir"


@pytest.fixture
def clear_temporary_dir():
    dir_path = Path(TMP_DIR)
    if dir_path.exists():
        shutil.rmtree(dir_path)


class TestCheckFile:
    def test_check_file_exist_file(self):
        file_path = "src/util.py"
        actual = util.check_file(file_path=file_path)
        assert actual

    def test_check_not_exist_file(self):
        file_path = "not_exist_file.txt"
        with pytest.raises(FileNotFoundError):
            _ = util.check_file(file_path=file_path)


class TestCheckDir:
    def test_check_dir_exist_dir(self):
        dir_path = "tools"
        actual = util.check_dir(dir=dir_path, mkdir=False)
        assert actual

    def test_check_dir_not_exist(self):
        dir_path = "test/not_exist_dir"
        with pytest.raises(FileNotFoundError):
            _ = util.check_dir(dir=dir_path, mkdir=False)

    def test_check_dir_mkdir(self, clear_temporary_dir):
        with pytest.raises(FileNotFoundError):
            _ = util.check_dir(dir=TMP_DIR, mkdir=False)
        dir_path = util.check_dir(dir=TMP_DIR, mkdir=True)
        assert dir_path.exists()
