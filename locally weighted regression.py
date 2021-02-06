import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv('./data.csv',
                   usecols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                            'MEDV'])
x = np.array(data.drop('MEDV',axis=1).values)
y = np.array(data['MEDV'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# def locally_weigh_lr(test_point, x, y, k):
#     x = np.mat(x)
#     y = np.mat(y).T
#     n = x.shape[0]
#     weights = np.mat(np.eye(n))
#
#     for i in range(n):
#         diff = test_point - x[i, :]
#         weights[i, i] = np.exp(diff * diff.T / (-2 * k ** 2))
#
#     XTX = x.T * (weights * x)
#
#     if np.linalg.det(XTX) == 0.0:
#         print('stopped')
#     else:
#         ws = XTX.I * (x.T * (weights * y))
#         return test_point * ws
#
#
# def locally_weight_lr_test(test, x, y, k):
#     n = x.shape[0]
#     y_hat = np.zeros(n)
#
#     for i in range(n):
#         y_hat[i] = locally_weigh_lr(test[i], x, y, k)
#
#     return y_hat
#
#
# y_hat = locally_weight_lr_test(x_train, x_train, y_train, 0.1)
# print(y_hat)
