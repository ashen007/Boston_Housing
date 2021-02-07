import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
