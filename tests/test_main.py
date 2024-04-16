# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from text_rec_metric import TextRecMetric

test_file_dir = cur_dir / "test_files"
metric = TextRecMetric()


def test_normal():
    pred_path = test_file_dir / "pred.txt"
    result = metric(pred_path)
    print(result)
    assert result["ExactMatch"] == 0.8323
    assert result["CharMatch"] == 0.9355


def test_input_none():
    with pytest.raises(ValueError) as exc_info:
        metric(None)

    assert exc_info.type is ValueError
