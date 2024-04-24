import numpy as np
import cv2


def cut_image(image: np.ndarray, box_coords: list) -> np.ndarray:
    """
    Входные данные:
    image_source: путь к файлу с изображением
    box_coords: лист из координат бокса

    Выходные данные:
    1) обрезанное изображение с областью со стропой
    """
    # image = cv2.imread(image_source)
    if box_coords == None:
        return image
    else:
        x1, y1, x2, y2 = box_coords
        return image[y1:y2, x1:x2]
