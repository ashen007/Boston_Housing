# gradient decent function
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

X = np.array([8, 7, 6, 3, 8, 5, 2, 1, 3, 9, 1, 2, 0, 9, 1, 2])
Y = np.array([0, 2, 3, 9, 8, 3, 0, 4, 3, 2, 8, 2, 3, 0, 4, 7, ])
coeff = 1
inter = 2
alpha = 0.001
epochs = 100


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


def visualize(x, y):
    plt.figure(figsize=[8, 8], dpi=200)
    sns.scatterplot(x, y)


def prediction(x, coefficient, intercept):
    return x * coefficient + intercept


def cost_function(x, y, coefficient, intercept):
    return y - (x * coefficient + intercept)


def main(x, y, coefficient, intercept, learning_rate, epochs):
    # call gradient decent as we needed (epochs)
    cost_history = []
    for i in range(epochs):
        coefficient, intercept = gradient_decent(x, y, coefficient, intercept, learning_rate)
        predictions = prediction(x,coefficient,intercept)
        cost_history.append(cost_function(x,y,coefficient,intercept))
        sns.lineplot(x,predictions,color='red')

    plt.show()
    return cost_history


if __name__ == '__main__':
    visualize(X, Y)
    cost_history = main(X, Y, coeff, inter, alpha, epochs)
    plt.figure(figsize=[12,12],dpi=200)
    plt.plot(cost_history,np.arange(0,epochs))
    plt.show()
