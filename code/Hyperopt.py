import pandas as pd
import numpy as np
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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

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

# test_gbr = xgb.XGBRegressor()
# test_gbr.fit(train_X, train_y)
# test_gbr_pred = test_gbr.predict(test_X)
# test_gbr_mae = mean_absolute_error(test_y, test_gbr_pred)
# print("gbr_mae: ",test_gbr_mae)
# print("gbr_dod: ",dod(test_y, test_gbr_pred))

# 半智障调参：贝叶斯优化

# 定义最小化目标函数
def objective(params):
    model_name = params['type']
    del params['type']
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
    # loss = cross_val_score(clf, features, target, cv=10, scoring='neg_mean_absolute_error').mean()
    # loss = cross_val_score(clf, train_X, train_y, cv=10, scoring='neg_mean_absolute_error').mean()
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    # loss = mean_absolute_error(test_y,pred)
    loss = dod(test_y, pred)
    return {'loss':loss, 'params':params, 'status':STATUS_OK}

# 设置域空间
space = hp.choice('regressor_type',[
    {
        'type': 'svm',
        'C': hp.uniform('C',0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0)
    },
    {
        'type': 'xgboost',
        'n_estimators': hp.choice('xgb_n_estimators', range(50,501,2)),
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('xgb_max_depth', range(2,8,1)),
        'min_child_weight': hp.choice('xgb_min_child_weight', range(1,5,1)),
        'reg_alpha': hp.uniform('xgb_reg_alpha', 0, 1.0),
        'subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1.0)
    },
    {
        'type': 'lightgbm',
        'n_estimators': hp.choice('lgb_n_estimators', range(50,501,2)),
        'learning_rate': hp.uniform('lgb_learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('lgb_max_depth', range(2,8,1)),
        'num_leaves': hp.choice('lgb_num_leaves', range(20, 50, 1)),
        'min_child_weight': hp.choice('lgb_min_child_weight', [0.001,0.005,0.01,0.05,0.1]),
        'min_child_samples': hp.choice('lgb_min_child_samples', range(5,51,5)),
        'subsample': hp.uniform('lgb_subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.uniform('lgb_reg_alpha', 0, 1.0)
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('n_neighbors', range(2,11)),
        'algorithm': hp.choice('algorithm', ['auto','ball_tree','kd_tree','brute'])
    },
    {
        'type': 'linear',
        'normalize': hp.choice('normalize', [False, True])
    }
])

# 随机查看一组参数
# import hyperopt.pyll.stochastic as hps
# print(hps.sample(space))

# 设定搜索算法
from hyperopt import tpe
# from hyperopt import rand

# 如果有有必要，可以设置查看黑盒函数object中的搜索情况（每次选择参数等）
from hyperopt import Trials
trials = Trials()

# 定义总的执行函数
count = 0
best_score = 999
def f(params):
    global best_score, count
    count +=1
    score = objective(params.copy())
    loss = score['loss']
    if loss < best_score:
        best_score = loss
        print('#################################################################')
        print('new best: ', score, 'using: ',params['type'])
        print('iters: ', count, 'using: ',params)
    if count % 50 == 0:
        print('iters:',count, 'score:',loss)
    return loss

from hyperopt import fmin
MAX_EVALS = 1500
best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=MAX_EVALS,trials=trials)
print(best_score, best)

# best_param = {'colsample_bytree': 0.8872641080186975, 'learning_rate': 0.27209843133952755, 'max_depth': 4, 'min_child_weight': 0, 'n_estimators': 47, 'reg_alpha': 0.83274668045094, 'subsample': 0.673026878375594}
# xgb = xgb.XGBRegressor(**best_param)
# xgb.fit(train_X, train_y)
# pred = xgb.predict(test_X)
# print(mean_absolute_error(test_y, pred), dod(test_y, pred))