# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from text_rec_metric import TextRecMetric

metric = TextRecMetric()

pred_path = "pred.txt"
metric = metric(pred_path)
print(metric)
