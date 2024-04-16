import numpy as np

class Point2D:
    def __init__(self, x=0., y=0.) -> None:
        self.x = x
        self.y = y
    
class Line:
    def __init__(self, x1=0., y1=0., x2=1., y2=1.) -> None:
        self.p1 = Point2D(x1, y1)
        self.p2 = Point2D(x2, y2)

    def k (self) -> float:
        '''Угол наклона вектора на плоскости'''
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    def b (self) -> float:
        '''Сдвиг вдоль горизонтальной оси'''
        return(self.p2.x * self.p1.y - self.p1.x * self.p2.y) / (self.p2.x - self.p1.x)

    def angle(self, line, type='rad') -> float:
        '''Угол между прямыми.
        - Парметр type: если type = "rad" (по умолчанию), то результат функции возвращается в радианах,
        если type = "grad", то результат будет переведён в градусы'''
        k1 = self.k()
        k2 = line.k()
        znam = (1. + k2 * k1)
        if znam == 0:
            if type == "rad":
                return np.pi
            elif type == "grad":
                return np.rad2deg(np.pi)

        rad = np.arctan((k2 - k1) / znam)
        if type == "rad":
            return rad
        elif type == "grad":
            return np.rad2deg(rad)

    def equation(self, order=10) -> None:
        '''Выводит в консоль уравнение прямой в формате y = kx + b
        - параметр order: число знаков после запятой при выводе'''
        print(f'y = {round(self.k(), order)} x + {round(self.b(), order)}')


def angles(lines, type='rad'):
    '''
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
    '''
    res = np.zeros((len(lines), len(lines)))
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            if (i > j):
                continue
            line1 = lines[i]
            line2 = lines[j]
            res[i][j] = line1.angle(line2, type)
            if type=="rad":
                res[j][i] = np.mod(2*np.pi- res[i][j], 2*np.pi)
            elif type == "grad":
                res[j][i] = np.mod(180. - res[i][j], 180)
    return res
