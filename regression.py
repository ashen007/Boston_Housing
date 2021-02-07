import numpy as np
import get_data
from sklearn.metrics import r2_score


def stand_reg(x, y):
    xMatrix = np.array(x)
    yMatrix = np.array(y).T
    XTX = np.dot(xMatrix.T, xMatrix)

    if np.linalg.det(XTX) == 0.0:
        print('can not calculate inverse')
        return
    ws = np.dot(np.linalg.inv(XTX), (np.dot(xMatrix.T, yMatrix)))
    return ws


df, target_feature = get_data.independent_data(get_data.get_data())
ws = stand_reg(df.values, target_feature.values)
predicts = np.dot(np.asarray(df), ws)
print(r2_score(target_feature, predicts))
