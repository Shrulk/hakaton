import numpy as np


class Point2D:
    def __init__(self, x=0.0, y=0.0) -> None:
        self.x = x
        self.y = y


class Line:
    def __init__(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0) -> None:
        self.p1 = Point2D(x1, y1)
        self.p2 = Point2D(x2, y2)

    def k(self) -> float:
        """Угол наклона вектора на плоскости"""
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    def b(self) -> float:
        """Сдвиг вдоль горизонтальной оси"""
        return (self.p2.x * self.p1.y - self.p1.x * self.p2.y) / (self.p2.x - self.p1.x)

    def angle(self, line, type="rad") -> float:
        """Угол между прямыми.
        - Парметр type: если type = "rad" (по умолчанию), то результат функции возвращается в радианах,
        если type = "grad", то результат будет переведён в градусы"""
        k1 = self.k()
        k2 = line.k()
        znam = 1.0 + k2 * k1
        if znam == 0:
            if type == "rad":
                return np.pi
            elif type == "grad":
                return np.rad2deg(np.pi)

        rad = np.arctan((k2 - k1) / znam) + np.pi / 2
        if type == "rad":
            return rad
        elif type == "grad":
            return np.rad2deg(rad)

    def equation(self, order=10) -> None:
        """Выводит в консоль уравнение прямой в формате y = kx + b
        - параметр order: число знаков после запятой при выводе"""
        print(f"y = {round(self.k(), order)} x + {round(self.b(), order)}")


def angles(lines, type="rad"):
    """
    Возвращает матрицу попарных углов между прямыми из входного массива.

    - Пример:\\
    line1 = Line(1, 2, 5, 6)\\
    line2 = Line(10, 1, 5, 6)\\
    line3 = Line(5, 8, 2, 2)\\
    line4 = Line(1, 1, 9, 2)\\
    line5 = Line(-10, 6, 4, 2)\\
    print(angles([line1, line2, line3, line4, line5], "rad"))\\
    Out:\\
    [[ 0.          3.14159265  0.32175055 -0.66104317 -1.06369782]\\
     [ 3.14159265  0.         -1.24904577  0.90975316  0.5070985 ]\\
     [ 5.96143475  1.24904577  0.         -0.98279372 -1.38544838]\\
     [ 0.66104317  5.37343215  0.98279372  0.         -0.40265465]\\
     [ 1.06369782  5.7760868   1.38544838  0.40265465  0.        ]]\\
    """
    res = np.zeros((len(lines), len(lines)))
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            if i > j:
                continue
            line1 = lines[i]
            line2 = lines[j]
            ang = intersection_point_and_angle(
                [
                    line1.p1.x,
                    line1.p1.y,
                    line1.p2.x,
                    line1.p2.y,
                    line2.p1.x,
                    line2.p1.y,
                    line2.p2.x,
                    line2.p2.y,
                ]
            )
            res[i][j] = ang
            res[j][i] = 180 - ang
            # if ang < ((np.pi / 2) if type == "rad" else 90):
            #     res[i][j] = (
            #         (np.mod(ang + 2 * np.pi, 2 * np.pi))
            #         if type == "rad"
            #         else np.mod(ang + 360, 2 * 180)
            #     )  # ang
            # else:
            #     res[i][j] = (
            #         (np.mod((np.pi - ang) + 2 * np.pi, 2 * np.pi))
            #         if type == "rad"
            #         else np.mod((180 - ang) + 360, 2 * 180)
            #     )
            # res[i][j] = np.pi - ang
            # if type == "rad":
            #     res[j][i] = np.pi - res[i][j]
            # elif type == "grad":
            #     res[j][i] = 180.0 - res[i][j]
    # print(res)
    return res


def find_angle(dots):
    if dots[0][1] > dots[1][0]:
        line_1 = np.array([dots[0], dots[1]])
    else:
        line_1 = np.array([dots[1], dots[0]])

    if dots[2][1] > dots[3][0]:
        line_2 = np.array([dots[2], dots[3]])
    else:
        line_2 = np.array([dots[3], dots[2]])

    vector_1 = line_1[0] - line_1[1]
    vector_2 = line_2[0] - line_2[1]
    return np.arccos(
        np.dot(vector_1, vector_2) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)
    )


def intersection_point_and_angle(points):
    # Проверяем, что переданы 8 координат точек
    if len(points) != 8:
        raise ValueError("Должно быть передано 8 координат точек")

    # Разбиваем точки на координаты X и Y для каждой прямой
    x1, y1, x2, y2, x3, y3, x4, y4 = points

    # Вычисляем уравнения прямых через две точки для каждой прямой
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x1 * (y1 - y2) - y1 * (x1 - x2)

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x3 * (y3 - y4) - y3 * (x3 - x4)

    # Находим точку пересечения прямых
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        raise ValueError("Прямые параллельны, нет точки пересечения")
    else:
        x_intersect = (b1 * c2 - b2 * c1) / determinant
        y_intersect = (a2 * c1 - a1 * c2) / determinant

    # Находим нижнюю точку на каждой прямой
    lower_point1 = (x1, y1) if y1 > y2 else (x2, y2)
    lower_point2 = (x3, y3) if y3 > y4 else (x4, y4)

    # Вычисляем угол между двумя нижними точками и точкой пересечения
    vector1 = np.array(lower_point1) - np.array([x_intersect, y_intersect])
    vector2 = np.array(lower_point2) - np.array([x_intersect, y_intersect])
    angle_rad = np.arccos(
        np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    )
    angle_deg = np.degrees(angle_rad)

    return angle_deg
