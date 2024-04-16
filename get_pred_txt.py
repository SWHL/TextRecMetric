# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

engine = RapidOCR()

dataset = load_dataset("SWHL/text_rec_test_dataset")
test_data = dataset["test"]

content = []
for i, one_data in enumerate(tqdm(test_data)):
    img = np.array(one_data.get("image"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result, elapse = engine(img, use_det=False, use_cls=False, use_rec=True)
    if result is None:
        rec_res = ""
        elapse = 0
    else:
        rec_res, elapse = result[0]

    gt = one_data.get("label", None)
    content.append(f"{rec_res}\t{gt}\t{elapse}")

with open("pred.txt", "w", encoding="utf-8") as f:
    for v in content:
        f.write(f"{v}\n")
