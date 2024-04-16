import typing
import numpy as np


def is_in_image(index: typing.Union[np.ndarray, list, tuple],
                shape: typing.Union[np.ndarray, list, tuple]):
    x = np.max([0, np.min([index[0], shape[0]])])
    y = np.max([0, np.min([index[1], shape[1]])])
    return x, y


def string_diagonal_definition(box: typing.Union[np.ndarray, list], gap: int) -> bool:
    diagonal_distribution_left = []
    diagonal_distribution_right = []
    box_shape = np.shape(box)
    i_s_left = np.arange(0, box_shape[0], 1)
    j_s_left = np.arange(0, box_shape[1], 1)
    indices_left = list(zip(i_s_left, j_s_left))
    list(map(lambda x: diagonal_distribution_left.append(box[x]), indices_left))

    i_s_right = np.arange(box_shape[0] - 1, -1, -1)
    j_s_right = np.arange(0, box_shape[1], 1)
    indices_right = list(zip(i_s_right, j_s_right))
    list(map(lambda x: diagonal_distribution_right.append(box[x]), indices_right))

    if np.std(diagonal_distribution_left) >= np.std(diagonal_distribution_right):
        return False
    else:
        return True



