import numpy as np
import get_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

data, target = get_data.independent_data(get_data.get_data())
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)
X = np.mat(train_x)
Y = np.mat(train_y)
t_sample = test_y.head(10)


def ridge_regression(x, y, alpha):
    xtx = np.dot(x.T, x)
    ridge = np.linalg.pinv(np.dot(alpha, np.eye(x.shape[1]))) + xtx
    return np.dot(ridge, (np.dot(x.T, y)))


def train(x, y):
    """normalize"""
    y = y.T
    alphas = np.arange(0, 1, 0.001)
    epochs = len(alphas)
    # x_mean = np.mean(x, axis=0)
    # x_variance = np.var(x, axis=0)
    # y_mean = np.mean(y, axis=0)
    # y = y - y_mean
    # x = (x - x_mean) / x_variance

    # ridge_weight calculation
    w = np.zeros((epochs, x.shape[1]))

    for i in range(epochs):
        # print(alphas[i])
        ws = ridge_regression(x, y, alphas[i])
        w[i, :] = ws.T

    return w


ridge_weights = train(X[:10], Y[:, :10])

fig = plt.figure(figsize=[8, 8], dpi=200)
ax = fig.add_subplot(111)
ax.plot(ridge_weights)
# plt.savefig('graphs/weights changing with alpha.jpg')
plt.show()


def prediction(x, ws):
    predicts = []

    for i in range(ridge_weights.shape[0]):
        predicts.append(np.dot(x, ws[i]))
    return predicts


predictions = prediction(test_x.head(10), ridge_weights)
print(predictions)

for i in range(len(predictions)):
    print(f"alpha{i}: r2 score {r2_score(t_sample, predictions[i])} MSE {mean_squared_error(t_sample, predictions[i])}")

# alphas = np.arange(0,1,0.01)
# for i in range(len(alphas)):
#     w = Ridge(alpha=alphas[i])
#     w.fit(train_x, train_y)
#     predict = w.predict(test_x)
#     print(r2_score(test_y, predict), mean_squared_error(test_y, predict))
