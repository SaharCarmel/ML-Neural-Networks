import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pb


class LinearRegression(object):
    def __init__(self, x, y, n):
        self.x_vec = x
        self.y_vec = y
        self.a = 0
        self.b = 0
        self.r_sqr = 0
        self.n_lin = 0
        self.yhat = 0
        self.dim = n

    def reg(self):

        d_lin = np.sum(self.x_vec)
        f_lin = np.sum(self.y_vec)
        c_lin = self.x_vec.dot(self.x_vec)
        e_lin = self.x_vec.dot(self.y_vec)
        self.n_lin = self.x_vec.shape[0]
        self.a = (f_lin * d_lin - self.n_lin * e_lin) / (d_lin ** 2 - self.n_lin * c_lin)
        self.b = (e_lin * d_lin - c_lin * f_lin) / (d_lin ** 2 - self.n_lin * c_lin)



    def plot(self):
        plt.scatter(self.x_vec, self.y_vec)
        self.yhat = self.a * self.x_vec + self.b
        plt.plot(self.x_vec, self.yhat)
        plt.show()
        print("a:", self.a, "b:", self.b, "r_squared", self.r_squared())

    def r_squared(self):
        d1 = self.y_vec - self.yhat
        d2 = self.y_vec - self.yhat.mean()
        self.r_sqr = 1 - d1.dot(d1)/d2.dot(d2)
        return self.r_sqr
    pass


def open_csv(str1):
    # Importing data using for loop, bot the best solution
    x_vec = []
    y_vec = []
    for line in open(str1):
        x, y = line.split(',')
        x_vec.append(float(x))
        y_vec.append(float(y))

    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)

    return x_vec, y_vec


def open_pb(str1):
    # Importing data using pandas
    file_fun = pb.read_csv(str1).as_matrix()
    x_vec = file_fun[:, 0]
    y_vec = file_fun[:, 1]
    return x_vec, y_vec


class TwoDLin(object):
    def __init__(self, str1):
        self.file_loc = str1
        self.x1 = 0
        self.x2 = 0
        self.y = 0
        self.X = []
        self.Y = []
        self.w = 0
        self.yhat = np.array([0])
        self.r_sqr = 0
        for line in open(self.file_loc):
            self.x1, self.x2, self.y = line.split(',')
            self.X.append([float(self.x1), float(self.x2), 1])  # add the bias term
            self.Y.append(float(self.y))

        # let's turn X and Y into numpy arrays since that will be useful later
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def regression(self):
        self.w = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y))
        self.yhat = np.dot(self.X, self.w)

    def r_squared(self):
        d1 = self.Y - self.yhat
        d2 = self.Y - self.yhat.mean()
        self.r_sqr = 1 - d1.dot(d1) / d2.dot(d2)
        return self.r_sqr


# z = TwoDLin('data_2d.csv')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(z.X[:, 0], z.X[:, 1], z.Y)
# plt.show()

z = TwoDLin('data_2d.csv')
z.regression()
print("r-squared:", z.r_squared())




