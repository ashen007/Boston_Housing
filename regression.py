import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_data():
    data = pd.read_csv('./data.csv',usecols=['CRIM','ZN','INDUS','CHAS','NOX','AGE',
                                             'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'], na_filter=True)
    vif = pd.DataFrame()

    # while True:
    #     vif['features'] = data.columns
    #     vif['vif value'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    #     vif.sort_values(by='vif value',inplace=True)
    #     vif.reset_index(inplace=True)
    #     if all(vif['vfi values'] < 10):
    #         break
    #     else:
    #         data.drop(vif.iloc[0]['features'],axis=1,inplace=True)

    return data


data = get_data()

vfi['features'] = data.columns
vfi['vfi value'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
vfi.sort_values(by='vfi value',inplace=True,ascending=False)

if all(vfi['vfi value'] < 10):
    print('all independent')
else:
    print('dependent features')
