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


def diagonal_elements_extraction(
    image: typing.Union[np.ndarray, list], gap: int, *args, **kwargs
) -> typing.Tuple[list, list]:
    diagonal_distribution_left = []
    diagonal_distribution_right = []
    image_shape = np.shape(image)
    i_s_left = np.arange(0, image_shape[0], 1)
    j_s_left = np.arange(0, image_shape[1], 1)

    i_s_right = np.arange(image_shape[0] - 1, -1, -1)
    j_s_right = np.arange(0, image_shape[1], 1)

    indices_left = np.array(list(zip(i_s_left, j_s_left)))
    indices_right = np.array(list(zip(i_s_right, j_s_right)))

    for number in range(gap):
        indices_left = np.concatenate(
            (
                np.copy(indices_left),
                np.copy(indices_left) + [0, number],
                np.copy(indices_left) + [number, 0],
                np.copy(indices_left) - [0, number],
                np.copy(indices_left) - [number, 0],
            )
        )
        indices_right = np.concatenate(
            (
                np.copy(indices_right),
                np.copy(indices_right) + [0, number],
                np.copy(indices_right) + [number, 0],
                np.copy(indices_right) - [0, number],
                np.copy(indices_right) - [number, 0],
            )
        )
    indices_left = set(map(lambda x: tuple(x), indices_left))
    indices_right = set(map(lambda x: tuple(x), indices_right))
    list(
        map(
            lambda x: diagonal_distribution_left.append(
                image[is_in_image(x, image_shape)]
            ),
            indices_left,
        )
    )
    list(
        map(
            lambda x: diagonal_distribution_right.append(
                image[is_in_image(x, image_shape)]
            ),
            indices_right,
        )
    )
    return diagonal_distribution_left, diagonal_distribution_right


def diagonal_max_mean_definition(
    box: typing.Union[np.ndarray, list], gap: int, *args, **kwargs
) -> bool:
    diagonal_distribution_left, diagonal_distribution_right = (
        diagonal_elements_extraction(image=box, gap=gap)
    )
    if np.mean(diagonal_distribution_left) >= np.mean(diagonal_distribution_right):
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
    image_x_derivative = cv.Scharr(image, cv.CV_64F, 1, 0)
    image_y_derivative = cv.Scharr(image, cv.CV_64F, 0, 1)
    image_edges = np.sqrt(image_x_derivative**2 + image_y_derivative**2)
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
    image: np.ndarray, box_coordinates: list, gap: int = 2, *args, **kwargs
) -> bool:
    image = cut_frames.cut_image(image=image, box_coords=box_coordinates)
    angle = sling_diagonal_definition(image=image, gap=gap)
    return angle
