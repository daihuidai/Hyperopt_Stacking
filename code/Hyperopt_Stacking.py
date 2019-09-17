import numpy as np
import pandas as pd
from stacking_test import StackingModels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from hyperopt import hp
import hyperopt.pyll.stochastic as hps
from hyperopt import tpe
from hyperopt import rand
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import fmin
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../data/df_real_measure_20190902_1.csv', index_col=0)
df = df[(df['weight'] < 100) & (df['source'] == 'picc')]
df = df.iloc[:,3:]

features = df.iloc[:,1:]
target = df.iloc[:,0]
scaler = StandardScaler()
features = scaler.fit_transform(features)
train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.25, random_state=999)

def dod(real_values, pred_values):
    dod_result = np.around((np.absolute(real_values - pred_values) / real_values).mean(), 4)
    # dod_result = 1 - dod_result
    return dod_result

# 半智障调参：贝叶斯优化

# 定义最小化目标函数
def objective(params):
    model_name = params['model_name']
    del params['model_name']
    if model_name == 'svm':
        clf = SVR(**params)
    elif model_name == 'xgboost':
        clf = xgb.XGBRegressor(**params)
    elif model_name == 'lightgbm':
        clf = lgb.LGBMRegressor(**params)
    elif model_name == 'knn':
        clf = KNeighborsRegressor(**params)
    elif model_name == 'linear':
        clf = LinearRegression(**params)
    else:
        return 0
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    loss = dod(test_y, pred)
    return {'loss':loss, 'params':params, 'status':STATUS_OK}


space_svm = {
    'model_name': 'svm',
    'C': hp.uniform('C',0, 10.0),
    'kernel': hp.choice('kernel', ['linear', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20.0)
}
space_knn = {
    'model_name': 'knn',
    'n_neighbors': hp.choice('n_neighbors', range(2,11)),
    'algorithm': hp.choice('algorithm', ['auto','ball_tree','kd_tree','brute'])
}
space_linear = {
    'model_name': 'linear',
    'normalize': hp.choice('normalize', [False, True])
}
space_xgboost = {
    'model_name': 'xgboost',
    'n_estimators': hp.choice('n_estimators', range(50,501,2)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.choice('max_depth', range(2,8,1)),
    'min_child_weight': hp.choice('min_child_weight', range(1,5,1)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1.0),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}
space_lightgbm = {
    'type': 'lightgbm',
    'n_estimators': hp.choice('n_estimators', range(50,501,2)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.choice('max_depth', range(2,8,1)),
    'num_leaves': hp.choice('num_leaves', range(20, 50, 1)),
    'min_child_weight': hp.choice('min_child_weight', [0.001,0.005,0.01,0.05,0.1]),
    'min_child_samples': hp.choice('min_child_samples', range(5,51,5)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1.0)
}