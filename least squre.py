# simple linear regression (least square)
import gradient_decent
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv('./data.csv',usecols=['CRIM','MEDV'])
x = np.array(data['CRIM'].values)
y = np.array(data['MEDV'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def mean_square(x, y):
    n = len(x)
    coff_ = ((np.sum(y) * np.sum(x ** 2)) - (np.sum(x) * np.sum(x * y))) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    inter_ = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)

    return coff_, inter_


def prediction(x, coefficient, intercept):
    return x * coefficient + intercept


if __name__ == '__main__':
    coefficient, intercept = mean_square(x_train, y_train)
    plt.figure(figsize=[8,8],dpi=200)
    sns.scatterplot(x_test,y_test)

    for i in range(100):
        coefficient,intercept = gradient_decent.gradient_decent(x_train,y_train,coefficient,intercept,0.001)
        predictions = prediction(x_test,coefficient,intercept)
        sns.lineplot(x_test,predictions,color='red')

    plt.savefig('uni-varient.jpg')
    plt.show()
    print(coefficient, intercept)
