import numpy as np
import get_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


data,target = get_data.independent_data(get_data.get_data())
print(data,target)

def ridge
