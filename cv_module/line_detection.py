import typing
import numpy as np
import cv2 as cv
from cv_module import cut_frames


def is_in_image(
    index: typing.Union[np.ndarray, list, tuple],
    shape: typing.Union[np.ndarray, list, tuple],
    *args,
    **kwargs
):
    x = np.max([0, np.min([index[0], shape[0] - 1])])
    y = np.max([0, np.min([index[1], shape[1] - 1])])
    return x, y


def rectangular_eye_crafting(shape, shift=0):
    matrix = np.zeros(shape)
    k = shape[1] / shape[0]
    b = shift
    if k >= 1:
        for i in range(shape[0]):
            index = np.min([int(k * i + b), shape[1] - 1])
            index_2 = np.min([int(k * i + b) + int(np.round(k, 0)), shape[1] - 1])
            matrix[i][index:index_2] = 1
    else:
        matrix = np.zeros((shape[1], shape[0]))
        k = shape[0] / shape[1]
        b = shift
        for i in range(shape[1]):
            index = np.min([int(k * i + b), shape[0] - 1])
            index_2 = np.min([int(k * i + b) + int(np.round(k, 0)), shape[0] - 1])
            matrix[i][index:index_2] = 1
        matrix = matrix.T
    return np.array(matrix != 0, dtype=int)


def diagonal_elements_extraction(
    image: typing.Union[np.ndarray, list], gap: int, *args, **kwargs
) -> typing.Tuple[list, list]:
    image_shape = np.shape(image)
    eye_l = rectangular_eye_crafting(image_shape)
    for i in range(gap):
        eye_l = (
            eye_l
            + rectangular_eye_crafting(image_shape, i)
            + rectangular_eye_crafting(image_shape, -i)
        )
    eye_l = np.array(eye_l != 0, dtype=int)
    eye_r = np.flip(eye_l, axis=1)
    diagonal_distribution_left = image * eye_l

    diagonal_distribution_right = image * eye_r
    return diagonal_distribution_left, diagonal_distribution_right


def diagonal_max_mean_definition(
    box: typing.Union[np.ndarray, list], gap: int, *args, **kwargs
) -> bool:
    diagonal_distribution_left, diagonal_distribution_right = (
        diagonal_elements_extraction(image=box, gap=gap)
    )
    if np.sum(diagonal_distribution_left) >= np.sum(diagonal_distribution_right):
        return True  # левый верхний угол
    else:
        return False  # правый верхний угол


# def harris_measure_calculation(image: typing.Union[np.ndarray, list], k: float, *args, **kwargs) -> np.ndarray:
#     image_x_derivative = cv.Sobel(image, cv.CV_64F, 1, 0)
#     image_y_derivative = cv.Sobel(image, cv.CV_64F, 0, 1)
#     dx_sqr = image_x_derivative**2
#     dy_sqr = image_y_derivative**2
#     dydx = image_x_derivative*image_y_derivative
#     determinant = dx_sqr*dy_sqr - dydx**2
#     trace = dx_sqr+dy_sqr
#     measure = determinant-k*trace
#     return measure


def sling_diagonal_definition(image: typing.Union[np.ndarray, list], *args, **kwargs):
    temp_image = cv.bilateralFilter(image, 25, 15, 50)
    image_edges = cv.Canny(temp_image, 240, 250, None, 3)
    angle = diagonal_max_mean_definition(image_edges, *args, **kwargs)
    return angle


# def sling_diagonal_definition_harris(image: typing.Union[np.ndarray, list], *args, **kwargs):
#     harris_matrix = harris_measure_calculation(image=image, *args, **kwargs)
#     diagonal_distribution_left, diagonal_distribution_right = diagonal_elements_extraction(image=harris_matrix,
#                                                                                            *args,
#                                                                                            **kwargs)
#     left_sum = len(np.where(diagonal_distribution_left < [0])[0])
#     right_sum = len(np.where(diagonal_distribution_right < [0])[0])
#     if left_sum >= right_sum:
#         return 'left'
#     else:
#         return 'right'


def sling_diagonal_definition_from_file(
    image: np.ndarray, box_coordinates: list, gap: int = 5, *args, **kwargs
) -> bool:
    image = cut_frames.cut_image(image=image, box_coords=box_coordinates)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    angle = sling_diagonal_definition(image=image, gap=gap)
    return angle
