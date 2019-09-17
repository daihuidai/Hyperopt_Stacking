import numpy as np
import pandas as pd
from stacking_test import StackingModels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from hyperopt import hp, STATUS_OK
import hyperopt.pyll.stochastic as hps
from hyperopt import tpe, rand
from hyperopt import Trials, fmin
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
    params_copy = params.copy()
    model_name = params_copy['model_name']
    del params_copy['model_name']
    if model_name == 'svm':
        clf = SVR(**params_copy)
    elif model_name == 'xgboost':
        clf = xgb.XGBRegressor(**params_copy)
    elif model_name == 'lightgbm':
        clf = lgb.LGBMRegressor(**params_copy)
    elif model_name == 'knn':
        clf = KNeighborsRegressor(**params_copy)
    elif model_name == 'linear':
        clf = LinearRegression(**params_copy)
    else:
        return 0
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    loss = dod(test_y, pred)
    return {'loss':loss, 'params':params_copy, 'status':STATUS_OK}


def create_space():
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
        'model_name': 'lightgbm',
        'n_estimators': hp.choice('n_estimators', range(50,501,2)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('max_depth', range(2,8,1)),
        'num_leaves': hp.choice('num_leaves', range(20, 50, 1)),
        'min_child_weight': hp.uniform('min_child_weight', 0.001, 0.2),
        'min_child_samples': hp.choice('min_child_samples', range(5,51,5)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1.0)
    }
    return [space_svm, space_knn, space_xgboost, space_lightgbm]

svm_params = {'kernel':['linear', 'rbf']}
knn_params = {'n_neighbors':range(2,11), 'algorithm':['auto','ball_tree','kd_tree','brute']}
xgboost_params = {'n_estimators':range(50,501,2), 'max_depth':range(2,8,1), 'min_child_weight':range(1,5,1)}
lightgbm_params = {'n_estimators':range(50,501,2), 'max_depth':range(2,8,1), 'num_leaves':range(20,50,1), 'min_child_samples':range(5,51,5)}
all_model_params = {'svm':svm_params, 'knn':knn_params, 'xgboost':xgboost_params, 'lightgbm':lightgbm_params}
spaces = create_space()

def find_params(spaces):
    model_params = {}
    MAX_EVALS = 100
    for space in spaces:
        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        model_params[space['model_name']] = best_params
    return model_params

result = find_params(spaces)
print(result)

def init_base_models(result):
    result['svm']['kernel'] = all_model_params['svm']['kernel'][result['svm']['kernel']]
    result['knn']['n_neighbors'] = all_model_params['knn']['n_neighbors'][result['knn']['n_neighbors']]
    result['knn']['algorithm'] = all_model_params['knn']['algorithm'][result['knn']['algorithm']]
    result['xgboost']['n_estimators'] = all_model_params['xgboost']['n_estimators'][result['xgboost']['n_estimators']]
    result['xgboost']['max_depth'] = all_model_params['xgboost']['max_depth'][result['xgboost']['max_depth']]
    result['xgboost']['min_child_weight'] = all_model_params['xgboost']['min_child_weight'][result['xgboost']['min_child_weight']]
    result['lightgbm']['n_estimators'] = all_model_params['lightgbm']['n_estimators'][result['lightgbm']['n_estimators']]
    result['lightgbm']['max_depth'] = all_model_params['lightgbm']['max_depth'][result['lightgbm']['max_depth']]
    result['lightgbm']['num_leaves'] = all_model_params['lightgbm']['num_leaves'][result['lightgbm']['num_leaves']]

    svm = SVR(**result['svm'])
    knn = KNeighborsRegressor(**result['knn'])
    xgboost = xgb.XGBRegressor(**result['xgboost'])
    lightgbm = lgb.LGBMRegressor(**result['lightgbm'])
    return [svm, knn, xgboost, lightgbm]

base_models = init_base_models(result)
meta_model = LinearRegression()
sm = StackingModels(base_models, meta_model, n_folds=10)
sm.fit(train_X, train_y)
linear_pred = sm.predict(test_X)
print(linear_pred)
print(dod(test_y, linear_pred))