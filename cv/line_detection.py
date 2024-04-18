import typing
import numpy as np
import cv2 as cv


def is_in_image(index: typing.Union[np.ndarray, list, tuple],
                shape: typing.Union[np.ndarray, list, tuple]):
    x = np.max([0, np.min([index[0], shape[0] - 1])])
    y = np.max([0, np.min([index[1], shape[1] - 1])])
    return x, y


def diagonal_max_mean_definition(box: typing.Union[np.ndarray, list], gap: int) -> bool:
    diagonal_distribution_left = []
    diagonal_distribution_right = []
    box_shape = np.shape(box)
    i_s_left = np.arange(0, box_shape[0], 1)
    j_s_left = np.arange(0, box_shape[1], 1)

    i_s_right = np.arange(box_shape[0] - 1, -1, -1)
    j_s_right = np.arange(0, box_shape[1], 1)

    indices_left = np.array(list(zip(i_s_left, j_s_left)))
    indices_right = np.array(list(zip(i_s_right, j_s_right)))
    for number in range(gap):
        indices_left = np.concatenate((np.copy(indices_left),
                                       np.copy(indices_left) + [0, number],
                                       np.copy(indices_left) + [number, 0],
                                       np.copy(indices_left) - [0, number],
                                       np.copy(indices_left) - [number, 0]))
        indices_right = np.concatenate((np.copy(indices_right),
                                        np.copy(indices_right) + [0, number],
                                        np.copy(indices_right) + [number, 0],
                                        np.copy(indices_right) - [0, number],
                                        np.copy(indices_right) - [number, 0]))
    indices_left = set(map(lambda x: tuple(x), indices_left))
    indices_right = set(map(lambda x: tuple(x), indices_right))
    list(map(lambda x: diagonal_distribution_left.append(box[is_in_image(x, box_shape)]), indices_left))
    list(map(lambda x: diagonal_distribution_right.append(box[is_in_image(x, box_shape)]), indices_right))

    if np.mean(diagonal_distribution_left) >= np.mean(diagonal_distribution_right):
        return np.array(list(indices_left))
    else:
        return np.array(list(indices_right))


def sling_diagonal_definition(image: typing.Union[np.ndarray, list], *args, **kwargs):
    image_x_derivative = cv.Scharr(image, cv.CV_64F, 1, 0)
    image_y_derivative = cv.Scharr(image, cv.CV_64F, 0, 1)
    image_edges = np.sqrt(image_x_derivative ** 2 + image_y_derivative ** 2)
    angle = diagonal_max_mean_definition(image_edges, *args, **kwargs)
    return angle

