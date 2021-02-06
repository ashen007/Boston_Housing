import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score


def get_data():
    data = pd.read_csv('./data.csv', usecols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE',
                                              'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'], na_filter=True)
    return data


def independent_data(data):
    target = data['MEDV']
    data = data.drop('MEDV', axis=1)
    while True:
        vfi = pd.DataFrame()
        vfi['features'] = data.columns
        vfi['vfi value'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vfi.sort_values(by='vfi value', inplace=True, ascending=False)
        vfi.reset_index(inplace=True)

        if all(vfi['vfi value'] < 10):
            # print('all independent')
            break
        else:
            # print('dependent features')
            # print(vfi)
            # print(f"drop: {vfi.iloc[0]['features']}")
            data.drop(vfi.iloc[0]['features'], axis=1, inplace=True)
    return data, target


def stand_reg(x, y):
    xMatrix = np.array(x)
    yMatrix = np.array(y).T
    XTX = np.dot(xMatrix.T, xMatrix)

    if np.linalg.det(XTX) == 0.0:
        print('can not calculate inverse')
        return
    ws = np.dot(np.linalg.inv(XTX), (np.dot(xMatrix.T, yMatrix)))
    return ws


df, target_feature = independent_data(get_data())
ws = stand_reg(df.values, target_feature.values)
predicts = np.dot(np.asarray(df),ws)
print(r2_score(target_feature,predicts))
