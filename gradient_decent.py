# gradient decent function
import numpy as np
import seaborn as sns
import get_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


data,target = get_data.independent_data(get_data.get())
train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.2,random_state=42)


def gradient_decent(x, y, coefficient, intercept, learning_rate):
    """change coefficient and intercept when each time call this function"""
    coefficient_derivative = 0
    intercept_derivative = 0

    # call for loop to iterate over x and given y values and
    # calculate partial derivatives of coefficient and intercept
    for i in range(len(x)):  ## can use length of x or y because both same
        coefficient_derivative += -2 * x[i] * (y[i] - (coefficient * x[i] + intercept))
        intercept_derivative += -2 * (y[i] - (coefficient * x[i] + intercept))

    # learning_rate act as a hyper-meter to normalize changing to
    # achieve optimum result
    coefficient -= (coefficient_derivative * learning_rate) / len(x)
    intercept -= (intercept_derivative * learning_rate) / len(x)

    return coefficient, intercept


def cost_function(X, Y, beta):
    cost = np.sum((X.dot(beta) - Y) ** 2) / 2 / len(Y)
    return cost


def batch_gradient_decent(X, Y, beta, learning_rate, epochs):
    cost_hist = np.ones(epochs)

    for i in range(epochs):
        hypothesis = np.dot(X, beta)
        loss = hypothesis - Y
        gradient = np.dot(X.T, loss) / len(Y)
        beta = beta - gradient * learning_rate
        cost_hist[i] = cost_function(X, Y, beta)

    return beta, cost_hist


def train(X, Y, learning_rate, epochs):
    m, n = X.shape
    X = np.array(X)
    Y = np.array(Y).flatten()
    X = np.hstack([np.ones((m,1)),X])
    print(X)
    beta = np.zeros((n+1))
    beta,cost = batch_gradient_decent(X,Y,beta,learning_rate,epochs)

    return beta,cost


b,c = train(train_x,train_y,0.0001,10001)
# def visualize(x, y):
#     plt.figure(figsize=[8, 8], dpi=200)
#     sns.scatterplot(x, y)
#
#
# def prediction(x, coefficient, intercept):
#     return x * coefficient + intercept
#
#
# def cost_function(x, y, coefficient, intercept):
#     return y - (x * coefficient + intercept)


# def main(x, y, coefficient, intercept, learning_rate, epochs):
#     # call gradient decent as we needed (epochs)
#     cost_history = []
#     for i in range(epochs):
#         coefficient, intercept = gradient_decent(x, y, coefficient, intercept, learning_rate)
#         predictions = prediction(x, coefficient, intercept)
#         cost_history.append(cost_function(x, y, coefficient, intercept))
#         sns.lineplot(x, predictions, color='red')
#
#     plt.savefig('gradiant decent.jpg')
#     plt.show()
#     return cost_history
#
#
# if __name__ == '__main__':
#     visualize(X, Y)
#     cost_history = main(X, Y, coeff, inter, alpha, epochs)
#     plt.figure(figsize=[12, 12], dpi=200)
#     plt.plot(cost_history, np.arange(0, epochs))
#     plt.savefig('cost history.jpg')
#     plt.show()
