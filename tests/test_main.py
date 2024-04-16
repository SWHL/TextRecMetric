# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from text_rec_metric import TextRecMetric

metric = TextRecMetric()


def test_normal():
    pred = ["Hello world!"]
    gt = ["Holly world!"]
    result = metric(pred, gt)

    assert result["ExactMath"] == 0.0
    assert result["CharMatch"] == 0.8182


def test_input_none():
    pred = None
    gt = ["Holly world!"]
    result = metric(pred, gt)

    assert result == "preds or gts must not be None."
