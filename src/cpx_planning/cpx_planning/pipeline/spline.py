"""Cubic-spline implementation copied from OpenCDA for reference conditioning."""

import bisect
import math

import numpy as np


class Spline:
    """One-dimensional cubic spline used by the copied OpenCDA reference conditioner."""

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)
        h = np.diff(x)
        self.a = [iy for iy in y]
        matrix_a = self.__calc_A(h)
        vector_b = self.__calc_B(h)
        self.c = np.linalg.solve(matrix_a, vector_b)
        for index in range(self.nx - 1):
            self.d.append((self.c[index + 1] - self.c[index]) / (3.0 * h[index]))
            self.b.append((self.a[index + 1] - self.a[index]) / h[index] - h[index] * (self.c[index + 1] + 2.0 * self.c[index]) / 3.0)

    def calc(self, value):
        if value < self.x[0] or value > self.x[-1]:
            return None
        index = self.__search_index(value)
        delta = value - self.x[index]
        return self.a[index] + self.b[index] * delta + self.c[index] * delta ** 2.0 + self.d[index] * delta ** 3.0

    def calcd(self, value):
        if value < self.x[0] or value > self.x[-1]:
            return None
        index = self.__search_index(value)
        delta = value - self.x[index]
        return self.b[index] + 2.0 * self.c[index] * delta + 3.0 * self.d[index] * delta ** 2.0

    def calcdd(self, value):
        if value < self.x[0] or value > self.x[-1]:
            return None
        index = self.__search_index(value)
        delta = value - self.x[index]
        return 2.0 * self.c[index] + 6.0 * self.d[index] * delta

    def __search_index(self, value):
        return bisect.bisect(self.x, value) - 1

    def __calc_A(self, h):
        matrix = np.zeros((self.nx, self.nx))
        matrix[0, 0] = 1.0
        for index in range(self.nx - 1):
            if index != self.nx - 2:
                matrix[index + 1, index + 1] = 2.0 * (h[index] + h[index + 1])
            matrix[index + 1, index] = h[index]
            matrix[index, index + 1] = h[index]
        matrix[0, 1] = 0.0
        matrix[self.nx - 1, self.nx - 2] = 0.0
        matrix[self.nx - 1, self.nx - 1] = 1.0
        return matrix

    def __calc_B(self, h):
        vector = np.zeros(self.nx)
        for index in range(self.nx - 2):
            vector[index + 1] = 3.0 * (self.a[index + 2] - self.a[index + 1]) / h[index + 1] - 3.0 * (self.a[index + 1] - self.a[index]) / h[index]
        return vector


class Spline2D:
    """Two-dimensional cubic spline copied from OpenCDA."""

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        values = [0]
        values.extend(np.cumsum(self.ds))
        return values

    def calc_position(self, value):
        return self.sx.calc(value), self.sy.calc(value)

    def calc_curvature(self, value):
        dx = self.sx.calcd(value)
        ddx = self.sx.calcdd(value)
        dy = self.sy.calcd(value)
        ddy = self.sy.calcdd(value)
        return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    def calc_yaw(self, value):
        return math.atan2(self.sy.calcd(value), self.sx.calcd(value))
