from __future__ import print_function, annotations

from math import sqrt
import builtins


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
                builtins.print('[', matrix[0], sep ='', end = '',  **kwargs)
            for idx, _ in enumerate(matrix[1:]):
                builtins.print('', _) if idx != len(matrix) - 2 else builtins.print('', _, end='', **kwargs)
            builtins.print(']')


    # for i in args:
    #     if isinstance(i, (list, tuple)):
    #         for matrix in i:
    #             if len(matrix) != 1:
    #                 builtins.print('[', matrix[0], sep='', end='', **kwargs)
    #                 for idx, _ in enumerate(matrix[1:]):
    #                     builtins.print('', _, sep=' ', end='' if idx != len(matrix) - 2 else ']', **kwargs)
    #                 builtins.print('')
    #     else:
    #         try:
    #             builtins.print(*i, **kwargs, sep = ' ')
    #         except:
    #             builtins.print(i, **kwargs, sep = ' ')


class InitError(Exception):

    def __init__(self, message):
        super().__init__(message)


# class MatrixMathError(Exception):
#
#     def __init__(self, message):
#         super().__init__(message)


class Matrix:

    @classmethod
    def matrix(cls, *args):
        return cls(*args)

    @classmethod
    def calculateMinor(cls, matrix, st_index, fi_index):
        return [row[:fi_index] + row[fi_index + 1:] for row in (matrix[:st_index] + matrix[st_index + 1:])]


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

                att_dic = dict(zip([_.__class__ if _.__class__ != int else float for _ in args ], args))
                try:
                    matrix, num  = att_dic[Matrix], att_dic[float]
                except KeyError:
                    raise ValueError("Can not multiply matrices")
                # # expanded     = [(([num] * 1 if i == 0 else [0]) + [0 for j in range(matrix.shape[1]-1)]) for i in range(matrix.shape[0])]
                # expanded = [[num if i == 0 else 0 for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]
                # pprint(expanded)
                return func(matrix, num)

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
        # if len(self.matrix) != 1:
        #     builtins.print('[', self.matrix[0], sep = '')
        #     for idx, _ in enumerate(self.matrix[1:]):
        #         builtins.print('', _) if idx != len(self.matrix) - 2 else builtins.print('', _, end = '')
        #     builtins.print(']')
        # else:
        #     builtins.print('[', self.matrix[0], ']', sep='')
        pprint(self.matrix)

        return "-------\n<class 'Matrix'>"

    @MATRIXMATHERROR
    def __add__(self, other: Matrix) -> Matrix:
        return Matrix([[i + j for i, j in zip(k, v)]
                       for k, v in zip(self.matrix,
                                       other.matrix)])
        # if self.shape == other.shape:
        #     return Matrix([[i + j for i, j in zip(k, v)]
        #                           for k, v in zip(self.matrix,
        #                                           other.matrix)])
        # else:
        #     raise MatrixMathError("Matrices should be same shape")

    @MATRIXMATHERROR
    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix([[i - j for i, j in zip(k, v)]
                       for k, v in zip(self.matrix,
                                       other.matrix)])
        # if self.shape == other.shape:
        #     return Matrix([[i - j for i, j in zip(k, v)]
        #                           for k, v in zip(self.matrix,
        #                                           other.matrix)])
        # else:
        #     raise MatrixMathError("Matrices should be same shape")

    @MATRIXMULTERROR
    def __mul__(self, other: Matrix | float | int) -> Matrix:

        try:

            return Matrix([[sum([i * j for (i, j) in zip(row1, row2)])
                            for row2 in other.transposition()]
                           for row1 in self.matrix])

        except AttributeError:
            return Matrix([[i * other for i in row1] for row1 in self.matrix])

        # if self.shape[1] == other.shape[0]:
        # prod = []
        # for row1 in self.matrix:
        #     intermediate_prod = []
        #     for row2 in other.transposition():
        #         intermediate_prod.append(sum([i * j for i,j in zip(row1, row2)]))
        #     prod.append(intermediate_prod)
        # return Matrix(prod)
        # else:
        #     raise MatrixMathError("Can't multiply matrices")

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

    @classmethod
    def vector(cls, *args):
        return cls(*args)

    @staticmethod
    def multiply(mul):
        def _unwrapper(*args):
            return mul(Vector(coords=args[0].transposition()), args[1])

        return _unwrapper

    def __init__(self, coords : list):

        # assert any([type(_) == int or type(_) == float if type(coords[0] == int) else type(coords[0][i]) == int or type(coords[0][i]) == float for i, _ in enumerate(coords)]), "Incorrect coordinates"

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
            # self.magnitude = sqrt(sum([i ** 2 for i in self.coords]))

    def __str__(self):
        # if len(self.matrix) != 1:
        #     builtins.print('[', self.matrix[0], sep='')
        #     for idx, _ in enumerate(self.matrix[1:]): builtins.print('', _) if idx != len(
        #         self.matrix) - 2 else builtins.print('', _, end='')
        #     builtins.print(']')
        # else:
        #     builtins.print('[', self.matrix[0], ']', sep='')

        pprint(self.matrix)

        return "-------\n<class 'Vector'>"

    def __add__(self, other: Vector) -> Vector:
        return Vector(coords=[j for i in Matrix.__add__(self, other).matrix for j in i])

    def __sub__(self, other: Vector) -> Vector:
        return Vector(coords=[j for i in Matrix.__add__(self, other).matrix for j in i])

    @multiply
    def __mul__(self, other: Matrix | Vector | int) -> Vector | float:
        try:
            other.matrix
            return Matrix.__mul__(self, other).matrix[0][0]
            # if self.transposition() != other.matrix:
            #     return Matrix.__mul__(self, other).matrix[0][0]
            # else:
            #     self.magnitude = sqrt(Matrix.__mul__(self, other).matrix[0][0])
            #     return self.magnitude
        except AttributeError:
            num = other
            return Vector.vector([i * num for i in self.matrix[0]])

    def __pow__(self, exp = 2):
        assert exp == 2, "Can't pow vector"

        return self.__mul__(self, self)

    def __floordiv__(self, other : Vector):

        matrix = Matrix(matrix=[[1, 1, 1], self.transposition()[0], other.transposition()[0]])
        return Vector(coords=
                      [(-1) ** i
                        * matrix.determinant(Matrix.calculateMinor(matrix.matrix, 0, i))
                       for i in range(3)
                       ])

    def __truediv__(self, other):

        return self * (1 / other)

    def unitVector(self):

        return self / ((self ** 2) ** 0.5)

class Point:

    def __init__(self,
                 mass     : int | float,
                 coords   : list       ,
                 velocity : list       ,
                 friction : int | float):

        types_dict = {mass: 'mass', friction: 'friction'}
        invalid_types = [t for t in types_dict if not isinstance(t, (int, float))]
        if invalid_types:
            invalid_type = invalid_types[0]
            var_name = types_dict[invalid_type]
            raise InitError(f"Not implemented -{var_name}- argument type: {invalid_type.__class__}")

        self.mass = mass
        # sorted([coords, velocity])[0] += [0] * abs(len(velocity) - len(coords))
        self.coords = Matrix(Matrix([coords]).transposition())
        self.velocity = Vector(velocity)
        self.friction = friction

    def __str__(self):

        return '\n'.join(
            [key + ' : ' + str(val)
             if type(val) == int or type(val) == float
             else key + ' : ' + ' '.join([str(_) for _ in val.matrix])
                                 for key,val in self.__dict__.items()])


class RigidBody(Point):

    def __init__(self, pointField : list):
        self.field  = pointField
        self.center = sum([point.mass * point.matrix for point in self.field]) / \
                      sum([point.mass for point in self.field])

    def bodyMovement(self, ):
        ...



x = Point(mass=10, coords=[3.4, 5.6, 7.8], velocity=[3,-5, 0], friction = 0.7)
# y = Point(mass='10', coords=[3.4, 5.6, 7.8], velocity=[3,-5, 0], friction = 0.7)

print(x)


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
