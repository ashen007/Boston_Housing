# simple linear regression (least square)
import gradient_decent
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt

data = pd.read_csv('./data.csv', na_filter=True)
x = np.array(data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'DIS', 'RAD', 'LSTAT']].values)
y = np.array(data['MEDV'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def least_square_error(x, y):
    n = len(x)
    inter_ = ((np.sum(y) * np.sum(x ** 2)) - (np.sum(x) * np.sum(x * y))) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    coff_ = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)

    return coff_, inter_


def LSE_model(x, y):
    X = np.array(x)
    X = np.hstack((X, np.ones((len(X), 1))))
    Y = np.array(y)
    weights = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))

    return weights


def prediction(x, coefficient, intercept):
    return x * coefficient + intercept


if __name__ == '__main__':
    X = np.array(x_train)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.array(y_train)

    # coefficient, intercept = least_square_error(x_train, y_train)
    # coff, inter = np.linalg.lstsq(X, Y)[0]
    # predict_1 = prediction(x_test, coefficient, intercept)
    # predict_2 = prediction(x_test, coff, inter)

    # multi varient
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    test_x = np.array(x_test)
    test_x = np.hstack([test_x, np.ones((len(test_x), 1))])
    predict_4 = np.dot(test_x, beta)

    beta_1 = LSE_model(x_train, y_train)
    predict_5 = np.dot(test_x, beta_1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    predict_3 = model.predict(x_test)

    # plt.figure(figsize=[8, 8], dpi=200)
    # sns.scatterplot(x_test, y_test, color='#0C154A')
    # sns.lineplot(x_test, predict_1, color='#F24405')
    # sns.lineplot(x_test, predict_2, color='#F21905')
    # plt.savefig('graphs/compare_two least squre functions.jpg')
    # plt.show()

    # print(
    #     f'coefficient:{coefficient}, intercept:{intercept}, score:{r2_score(y_test, predict_1) * 100},'
    #     f' mse:{mean_squared_error(y_test, predict_1)}')
    # print(
    #     f'coefficient:{coff},intercept:{inter}, score:{r2_score(y_test, predict_2) * 100}, '
    #     f'mse:{mean_squared_error(y_test, predict_2)}')
    print(
        f'coefficient:{model.coef_},intercept:{model.intercept_}, score:{r2_score(y_test, predict_3) * 100}, '
        f'mse:{mean_squared_error(y_test, predict_3)}')
    print(
        f'beta values:{beta}, score:{r2_score(y_test, predict_4) * 100}, '
        f'mse:{mean_squared_error(y_test, predict_4)}')
    print(
        f'beta values:{beta_1}, score:{r2_score(y_test, predict_5) * 100}, '
        f'mse:{mean_squared_error(y_test, predict_5)}')
