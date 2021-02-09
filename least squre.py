# simple linear regression (least square)
import gradient_decent
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

data = pd.read_csv('./data.csv',usecols=['CRIM','MEDV'])
x = np.array(data['CRIM'].values)
y = np.array(data['MEDV'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def mean_square(x, y):
    n = len(x)
    inter_ = ((np.sum(y) * np.sum(x ** 2)) - (np.sum(x) * np.sum(x * y))) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    coff_ = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)

    return coff_, inter_


def prediction(x, coefficient, intercept):
    return x * coefficient + intercept


if __name__ == '__main__':
    X = np.array(x_train)
    X = np.vstack([X,np.ones(len(X))]).T
    Y = np.array(y_train)
    coefficient, intercept = mean_square(x_train, y_train)
    coff,inter = np.linalg.lstsq(X,Y)[0]
    predict_1 = prediction(x_test,coefficient,intercept)
    predict_2 = prediction(x_test,coff,inter)
    model = LinearRegression()
    model.fit(x_train.reshape(-1,1),y_train)
    predict_3 = model.predict(x_test.reshape(-1,1))

    plt.figure(figsize=[8,8],dpi=200)
    sns.scatterplot(x_test,y_test,color='#0C154A')
    sns.lineplot(x_test,predict_1,color='#F24405')
    sns.lineplot(x_test,predict_2,color='#F21905')
    plt.savefig('graphs/compare_two least squre functions.jpg')
    plt.show()

    print(f'coefficient:{coefficient}, intercept:{intercept}, score:{r2_score(y_test,predict_1)*100}, mse:{mean_squared_error(y_test,predict_1)}')
    print(f'coefficeint:{coff},intercept:{inter}, score:{r2_score(y_test,predict_2)*100}, mse:{mean_squared_error(y_test,predict_2)}')
    print(f'coefficeint:{model.coef_},intercept:{model.intercept_}, score:{r2_score(y_test,predict_3)*100}, mse:{mean_squared_error(y_test,predict_3)}')
