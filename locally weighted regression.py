import numpy as np
import get_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data, target = get_data.independent_data(get_data.get_data())
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
X = np.mat(x_test['AGE'])
X_lrg = np.mat(x_train['AGE'])
Y = np.mat(y_test)
Y_lrg = np.mat(y_train)

one = np.ones((1, Y.shape[1]))
X = np.hstack((one.T, X.T))
one_lrg = np.ones((1, Y_lrg.shape[1]))
X_lrg = np.hstack((one_lrg.T, X_lrg.T))


# function to calculate W weight diagnal Matric used in calculation of predictions
def kernel(point, train, k):
    n = train.shape[0]
    W = np.mat(np.eye(n))

    for i in range(n):
        diff = point - train[i]
        W[i, i] = np.exp((np.dot(diff, diff.T) / (-2 * k ** 2)))
    return W


def local_weights(point, x, y, k):
    weight = kernel(point, x, k)
    # use np.linalg.pinv otherwise when det is 0 no inverse
    W = np.linalg.pinv((x.T * (weight * x))) * (x.T * weight * y.T)
    return W


def localWeightRegression(x, y, k):
    m, n = np.shape(x)
    pred = np.zeros(m)

    for i in range(m):
        pred[i] = x[i] * local_weights(x[i], x, y, k)

    return pred


def stand_reg(x, y):
    y = y.T
    XTX = np.dot(x.T, x)

    if np.linalg.det(XTX) == 0.0:
        print('can not calculate inverse')
        return
    ws = np.dot(np.linalg.inv(XTX), (np.dot(x.T, y)))
    return ws


ws_lrg = stand_reg(X_lrg, Y_lrg)
one = np.ones((len(x_test), 1))
x = np.hstack((one, np.mat(x_test.AGE).T))
prediction_lrg = np.dot(x, ws_lrg)
print(r2_score(y_test.values, prediction_lrg) * 100)

predictions = localWeightRegression(X, Y, 0.5)
print(r2_score(y_test, predictions) * 100)

plt.figure(figsize=[8, 8], dpi=200)
plt.scatter(x_test['AGE'], y_test, color='#060E26', edgecolors='#DFE4F2')
plt.plot(X, predictions, color='#F29F05')
plt.plot(x_test['AGE'], prediction_lrg, color='#8C0D0D')
plt.savefig('graphs/LWR-vs-SLR.png')
plt.show()

# p = localWeightRegression(X, train_Y, 0.1)
# print(p)

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


# y_hat = locally_weight_lr_test(x_train, x_train, y_train, 0.1)
# print(y_hat)
