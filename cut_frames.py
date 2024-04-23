import numpy as np
from typing import Tuple, Union
import cv2
import json

def cut_image(image_source: str,
              boxes_source: str) -> Tuple[np.ndarray, Union[int, None], Union[int, None]]:
    """
    входные данные:
    image_source: путь к файлу с изображением
    boxes_source: путь к файлу .json с боксами для одного изображения
    
    выходные данные:
    1) обрезанное изображение от начала контейнера до верха изображения
    2) значения по координатам x и y для обратного перехода к координатам полного изображения из обрезанного
    """
    image = cv2.imread(image_source)
    with open(boxes_source) as json_file:
        result = json.load(json_file)
    if result['labels'] == None:
        return image, None, None
    else:
        box_index = result['labels'].index('box')
        box = result['boxes'][box_index]
        x1, y1, x2, y2 = box
        return image[:y2, x1:x2], x1, y1
    