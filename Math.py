from __future__ import print_function, annotations

from math import sqrt
import builtins


def sum(seq, start=0):
    if all([isinstance(_, Matrix) for _ in seq]):
        return builtins.sum(seq[1:], start=seq[0])
    else:
        return builtins.sum(seq, start)


def pprint(*args, **kwargs):
    """ Modifier for builtin function print to pretty print arrays aka matrices

    Typical use:

        m : Matrix = [[1,2], [3,4], [5,6]]
        pprint(m) -> [[1, 2]
                      [3, 4]
                      [5, 6]]

        pprint(10) -> 10

    :param args: list arguments we want to print beautifully
    :param kwargs: key word arguments for basic -builtin- print
    :return: pretty printed array # pprint(m : Matrix) -> ppa | pprint(a : Str|Float|Int) -> builtin.print
    """

    if any([i for i in args if type(i) == list or type(i) == tuple]):
        lManifold = [i for i in args if type(i) == list or type(i) == tuple]
        for matrix in lManifold:
            if len(matrix) != 1:
                builtins.print('[', matrix[0], sep='', **kwargs)
            else:
                builtins.print('[', matrix[0], sep='', end='', **kwargs)
            for idx, _ in enumerate(matrix[1:]):
                builtins.print('', _) if idx != len(matrix) - 2 else builtins.print('', _, end='', **kwargs)
            builtins.print(']')


class InitError(Exception):

    def __init__(self, message):
        super().__init__(message)



class Matrix:

    # m = Matrix([[1,2,3], [4,5,6], [7,8,9]]) -> [[1,2,3], [4,5,6], [7,8,9]]

    @classmethod
    def matrix(cls, *args):
        return cls(*args)

    @classmethod
    def calculateMinor(cls, matrix, stIndex, fiIndex):
        return [row[:fiIndex] + row[fiIndex + 1:] for row in (matrix[:stIndex] + matrix[stIndex + 1:])]

    @staticmethod
    def DETERMINANTERROR(func):
        def _unwrapper(*args):
            assert all([i == j for i, j in zip(args[0].shape, args[0].shape[1:])]), "Matrix should be square"
            return func(args[0], matrixMinor=args[1])

        return _unwrapper

    @staticmethod
    def MATRIXMATHERROR(func):
        def _unwrapper(*args):
            assert args[0].shape == args[1].shape, "Matrices should be same shape"
            return func(args[0], args[1])

        return _unwrapper

    @staticmethod
    def MATRIXMULTERROR(func):
        def _unwrapper(*args):
            try:
                assert args[0].shape[1] == args[1].shape[0], "Can't multiply matrices"
                return func(args[0], args[1])
            except AttributeError:

                att_dic = dict(zip([_.__class__ if _.__class__ != int else float for _ in args], args))
                try:
                    matrix, num = att_dic[Matrix], att_dic[float]
                except KeyError:
                    raise ValueError("Can not multiply matrices")

                return func(matrix, num)

        return _unwrapper

    @staticmethod
    def MATRIXDIVERROR(func):
        def _unwrapper(*args):
            assert isinstance(args[0], (Matrix, Vector)), "Division is not supported for matrices"
            return func(*args)

        return _unwrapper

    def __init__(self, matrix: list) -> None:
        if all([len(i) == len(matrix[0]) for i in matrix]):
            self.matrix = matrix
            self.shape = [len(matrix), len(matrix[0])]
        else:
            raise InitError("Cannot initialize matrix :: Inhomogeneous shape")

    def transposition(self) -> list:

        return [[self.matrix[j][i] for j in range(len(self.matrix))]
                for i in range(len(self.matrix[0]))]

    def __str__(self):

        pprint(self.matrix)

        return "-------\n<class 'Matrix'>"

    @MATRIXMATHERROR
    def __add__(self, other: Matrix) -> Matrix:
        return Matrix([[i + j for i, j in zip(k, v)]
                       for k, v in zip(self.matrix,
                                       other.matrix)])

    @MATRIXMATHERROR
    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix([[i - j for i, j in zip(k, v)]
                       for k, v in zip(self.matrix,
                                       other.matrix)])

    @MATRIXMULTERROR
    def __mul__(self, other: Matrix | float | int) -> Matrix:

        try:

            return Matrix([[sum([i * j for (i, j) in zip(row1, row2)])
                            for row2 in other.transposition()]
                           for row1 in self.matrix])

        except AttributeError:
            return Matrix([[i * other for i in row1] for row1 in self.matrix])

    @MATRIXDIVERROR
    def __truediv__(self, other):

        return self * (1 / other)

    @DETERMINANTERROR
    def determinant(self, matrixMinor: list) -> float:

        if len(matrixMinor) == 2:
            return matrixMinor[0][0] * matrixMinor[1][1] - \
                matrixMinor[0][1] * matrixMinor[1][0]
        elif len(matrixMinor) == 1:
            return matrixMinor[0][0]
        else:
            determinant = 0
            for row in range(len(matrixMinor)):
                determinant += ((-1) ** row) * matrixMinor[0][row] * \
                               self.determinant(Matrix.calculateMinor(matrixMinor, 0, row))
            return determinant


class Vector(Matrix):

    # v = Vector([1,2,3]) -> [[1], [2], [3]]

    @classmethod
    def vector(cls, *args):
        return cls(*args)

    @staticmethod
    def multiply(mul):
        def _unwrapper(*args):
            return mul(Vector(coords=args[0].transposition()), args[1])

        return _unwrapper

    def __init__(self, coords: list):

        if len(coords) == 1:
            if type(coords[0]) == list or type(coords[0]) == int or type(coords[0]) == float:
                pass
            else:
                raise InitError("Cannot initialize vector : Inhomogeneous shape")
        else:
            for _ in coords:
                if type(_) == int or type(_) == float:
                    pass
                else:
                    raise InitError("Cannot initialize vector : Inhomogeneous shape")

        coords = [[i] if type(coords[0]) != list else i for i in coords]
        self.matrix = coords
        try:
            self.shape = [len(self.matrix), len(self.matrix[0])]
        except KeyError:
            print('please, notice that vector should have coordinates')
            self.matrix = list(map(float, input('>>> ').split(' ')))

    def __str__(self):

        pprint(self.matrix)

        return "-------\n<class 'Vector'>"

    def __add__(self, other: Vector) -> Vector:
        return Vector(coords=[j for i in Matrix.__add__(self, other).matrix for j in i])

    def __sub__(self, other: Vector) -> Vector:
        return Vector(coords=[j for i in Matrix.__sub__(self, other).matrix for j in i])

    @multiply
    def __mul__(self, other: Matrix | Vector | int) -> Vector | float:
        try:
            other.matrix
            return Matrix.__mul__(self, other).matrix[0][0]

        except AttributeError:
            num = other
            return Vector.vector([i * num for i in self.matrix[0]])

    def __pow__(self, exp=2):
        assert exp == 2, "Can't pow vector"

        return self.__mul__(self, self)

    def __floordiv__(self, other: Vector):

        matrix = Matrix(matrix=[[1, 1, 1], self.transposition()[0], other.transposition()[0]])
        return Vector(coords=
                      [(-1) ** i
                       * matrix.determinant(Matrix.calculateMinor(matrix.matrix, 0, i))
                       for i in range(3)
                       ])

    def unitVector(self):

        return self / ((self ** 2) ** 0.5)


class Point:

    def __init__(self,
                 mass: int | float,
                 coords: list,
                 velocity: list,
                 friction: int | float):

        typDict = {mass: 'mass', friction: 'friction'}
        invDict = [t for t in typDict if not isinstance(t, (int, float))]
        if invDict:
            invType = invDict[0]
            varName = typDict[invType]
            raise InitError(f"Not implemented -{varName}- argument type: {invType.__class__}")

        self.mass = mass
        if not isinstance(coords, Matrix):
            # self.coords = Matrix(Matrix([coords]).transposition())
            self.coords = Vector(coords)
        else:
            self.coords = coords
        if not isinstance(velocity, Vector):
            self.velocity = Vector(velocity)
        else:
            self.velocity = velocity
        self.friction = friction

    def __str__(self):
        return '\n'.join(
            [key + ' : ' + str(val)
             if type(val) == int or type(val) == float
             else key + ' : ' + ' '.join([str(_) for _ in val.matrix])
             for key, val in self.__dict__.items()])

    def distance(self, other : Point) -> float:

        return sum([_[0]**2 for _ in (self.coords - other.coords).matrix]) ** 0.5


class RigidBody:

    def __init__(self, pointField: list):
        self.field    = pointField
        self.center   = Point(
                            mass     = sum([point.mass for point in self.field]),
                            coords   = sum([point.coords * point.mass for point in self.field])   / \
                                       sum([point.mass for point in self.field]),
                            velocity = sum([point.velocity * point.mass for point in self.field]) / \
                                       sum([point.mass for point in self.field]),
                            friction = sum([point.mass for point in self.field])                  / \
                                       sum([     1     for point in self.field]),
        )

        for point in pointField:
            centerUnit = point.velocity - self.center.coords
            point.lvelocity = centerUnit.unitVector() * (point.velocity * centerUnit)
            point.fvelocity = point.velocity - point.lvelocity
            # print(centerUnit, point.velocity, point.lvelocity, point.fvelocity, sep = '\n', end = '\n ======= \n')

        self.inertia = sum([point.mass * (point.distance(self.center))**2 for point in self.field])

    def bodyMovement(self):
        ...


x = Point(mass=10, coords=[3.4, 5.3, 0], velocity=[3, -5, 0], friction=0.7)
y = Point(mass=3, coords=[2.2, 0.7, 0], velocity=[3, -5, 0], friction=0.7)

b = RigidBody([x, y])

# print(x)

#
# class point:
#
#     def __init__(self,
#                  velocity=1, linear_velocity=0,
#                  angle=0, friction_koeff=0,
#                  mass=1, coord=[]):
#         self.velocity = velocity
#         self.linear_velocity = linear_velocity
#         self.angle = angle
#         self.mass = mass
#         self.friction = friction_koeff
#
#
# # apply initial velocity
#
# for y in body:
#     for x in y:
#         # pair : [y][x]
#         globals()['point_yx'] = point()
#
#
# def center_mass_calculate(body, point_a, point_b):
#
#     import numpy as np
#     from math import sqrt
#
#     koeff_c = 0.7
#
#     mass = sum(sum(np.array(body)))
#     obj_x, obj_y = 0, 0
#
#     for y in range(len(body)):
#         for x in range(len(y)):
#             obj_x += body[y][x] * x
#             obj_y += body[y][x] * y
#
#     x_cord     = obj_x / mass
#     y_cord     = obj_y / mass
#
#     mass_c = body[y_cord][x_cord]
#
#     velocity_a = point_a.velocity
#     velocity_b = point_b.velocity
#
#     cos_ab     = point_a.angle + point_b.angle
#
#     velocity_c = sqrt(velocity_a ** 2 + velocity_b ** 2 - 2 * velocity_a * velocity_b * cos_ab)
#     linear_c   = ...
#
#     mass_center = point(velocity=velocity_c,
#                     linear_velocity=linear_c,
#                     angle= 0,
#                     friction_koeff=koeff_c,
#                     mass=mass_c,
#                     coords=[x_cord, y_cord])
#
# if __name__ == "__main__":
#     velocity_a = 1
#     linear_a = 5
#     angle_a = 30
#     koeff_a = 0.7
#     mass_a = 1
#
#     velocity_b = -2
#     linear_b = -4
#     angle_b = 150
#     koeff_b = 0.7
#     mass_b = 1
#
#     point_a = point(velocity=velocity_a,
#                     linear_velocity=linear_a,
#                     angle=angle_a,
#                     friction_koeff=koeff_a,
#                     mass=mass_a,
#                     coords=[5, 5])
#
#     point_b = point(velocity=velocity_b,
#                     linear_velocity=linear_b,
#                     angle=angle_b,
#                     friction_koeff=koeff_b,
#                     mass=mass_b,
#                     coords=[10, 20])
#
