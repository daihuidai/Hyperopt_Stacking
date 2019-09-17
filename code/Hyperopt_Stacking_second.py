import numpy as np
import pandas as pd
from stacking_test import StackingModels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from hyperopt import hp, STATUS_OK
from hyperopt import tpe, rand, anneal
from hyperopt import Trials, fmin
import warnings

warnings.filterwarnings('ignore')

def obtain_data(file_path):
    global train_X,test_X,train_y,test_y
    df = pd.read_csv(file_path, index_col=0)
    df = df[(df['weight'] < 100) & (df['source'] == 'picc')]
    df = df.iloc[:,3:]
    features = df.iloc[:,1:]
    features = features[['body_l','body_h','heart_girth','chest_waist_d']]
    target = df.iloc[:,0]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.25)
    # return train_X, test_X, train_y, test_y

def dod(real_values, pred_values):
    dod_result = np.around((np.absolute(real_values - pred_values) / real_values).mean(), 4)
    # dod_result = 1 - dod_result
    return dod_result

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
    elif model_name == 'randomforest':
        clf = RandomForestRegressor(**params_copy)
    elif model_name == 'linear':
        clf = LinearRegression(**params_copy)
    elif model_name == 'gpr':
        clf = GaussianProcessRegressor(**params_copy)
    else:
        return 0
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    loss = dod(test_y, pred)
    return {'loss':loss, 'params':params_copy, 'status':STATUS_OK}

def create_space(list_name):
    spaces = []
    if 'svm' in list_name:
        space_svm = {
            'model_name': 'svm',
            'C': hp.uniform('C',0, 10.0),
            'kernel': hp.choice('kernel', ['linear', 'rbf']),
            'gamma': hp.uniform('gamma', 0, 20.0)
        }
        spaces.append(space_svm)
    if 'knn' in list_name:
        space_knn = {
            'model_name': 'knn',
            'n_neighbors': hp.choice('n_neighbors', range(1,14)),
            'algorithm': hp.choice('algorithm', ['auto','ball_tree','kd_tree','brute'])
        }
        spaces.append(space_knn)
    if 'xgboost' in list_name:
        space_xgboost = {
            'model_name': 'xgboost',
            'n_estimators': hp.choice('n_estimators', range(50,501,2)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(2,11,1)),
            'min_child_weight': hp.choice('min_child_weight', range(1,7,1)),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1.0),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }
        spaces.append(space_xgboost)
    if 'lightgbm' in list_name:
        space_lightgbm = {
            'model_name': 'lightgbm',
            'n_estimators': hp.choice('n_estimators', range(50,501,2)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(2,11,1)),
            'num_leaves': hp.choice('num_leaves', range(20, 61, 1)),
            'min_child_weight': hp.uniform('min_child_weight', 0.001, 0.2),
            'min_child_samples': hp.choice('min_child_samples', range(5,51,2)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1.0)
        }
        spaces.append(space_lightgbm)
    if 'randomforest' in list_name:
        space_randomforest = {
            'model_name': 'randomforest',
            'n_estimators': hp.choice('n_estimators', range(50,501,2)),
            'max_depth': hp.choice('max_depth', range(1,11,1)),
            'min_samples_split': hp.choice('min_samples_split', range(2,21,1)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1,21,1)),
            'max_features': hp.uniform('max_features', 0.4, 1.0)
        }
        spaces.append(space_randomforest)
    if 'linear' in list_name:
        space_linear = {
            'model_name': 'linear'
        }
        spaces.append(space_linear)
    if 'gpr' in list_name:
        space_gpr = {
            'model_name': 'gpr',
            'kernel': C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10)),
            'alpha': hp.uniform('alpha', 0.05, 1.0),
            'normalize_y': True
        }
        spaces.append(space_gpr)
    return spaces

def find_params(spaces, MAX_EVALS=1000):
    model_params = {}
    for space in spaces:
        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        model_params[space['model_name']] = best_params
    return model_params

def transform_choice(result,list_name):
    if 'svm' in list_name:
        svm_params = {'kernel':['linear', 'rbf']}
        result['svm']['kernel'] = svm_params['kernel'][result['svm']['kernel']]
    if 'knn' in list_name:
        knn_params = {'n_neighbors':range(1,14), 'algorithm':['auto','ball_tree','kd_tree','brute']}
        result['knn']['n_neighbors'] = knn_params['n_neighbors'][result['knn']['n_neighbors']]
        result['knn']['algorithm'] = knn_params['algorithm'][result['knn']['algorithm']]
    if 'xgboost' in list_name:
        xgboost_params = {'n_estimators':range(50,501,2), 'max_depth':range(2,11,1), 'min_child_weight':range(1,7,1)}
        result['xgboost']['n_estimators'] = xgboost_params['n_estimators'][result['xgboost']['n_estimators']]
        result['xgboost']['max_depth'] = xgboost_params['max_depth'][result['xgboost']['max_depth']]
        result['xgboost']['min_child_weight'] = xgboost_params['min_child_weight'][result['xgboost']['min_child_weight']]
    if 'lightgbm' in list_name:
        lightgbm_params = {'n_estimators':range(50,501,2), 'max_depth':range(2,11,1), 'num_leaves':range(20,61,1), 'min_child_samples':range(5,51,2)}
        result['lightgbm']['n_estimators'] = lightgbm_params['n_estimators'][result['lightgbm']['n_estimators']]
        result['lightgbm']['max_depth'] = lightgbm_params['max_depth'][result['lightgbm']['max_depth']]
        result['lightgbm']['num_leaves'] = lightgbm_params['num_leaves'][result['lightgbm']['num_leaves']]
    if 'randomforest' in list_name:
        randomforest_params = {'n_estimators': range(50,501,2), 'max_depth': range(1,11,1), 'min_samples_split': range(2,21,1), 'min_samples_leaf': range(1,21,1)}
        result['randomforest']['n_estimators'] = randomforest_params['n_estimators'][result['randomforest']['n_estimators']]
        result['randomforest']['max_depth'] = randomforest_params['max_depth'][result['randomforest']['max_depth']]
        result['randomforest']['min_samples_split'] = randomforest_params['min_samples_split'][result['randomforest']['min_samples_split']]
        result['randomforest']['min_samples_leaf'] = randomforest_params['min_samples_leaf'][result['randomforest']['min_samples_leaf']]
    if 'linear' in list_name:
        pass
    if 'gpr' in list_name:
        pass
    return result

def init_base_models(choice_model_params, list_name):
    base_models = []
    if 'svm' in list_name:
        svm = SVR(**choice_model_params['svm'])
        base_models.append(svm)
    if 'knn' in list_name:
        knn = KNeighborsRegressor(**choice_model_params['knn'])
        base_models.append(knn)
    if 'xgboost' in list_name:
        xgboost = xgb.XGBRegressor(**choice_model_params['xgboost'])
        base_models.append(xgboost)
    if 'lightgbm' in list_name:
        lightgbm = lgb.LGBMRegressor(**choice_model_params['lightgbm'])
        base_models.append(lightgbm)
    if 'randomforest' in list_name:
        randomforest = RandomForestRegressor(**choice_model_params['randomforest'])
        base_models.append(randomforest)
    if 'linear' in list_name:
        linear = LinearRegression(**choice_model_params['linear'])
        base_models.append(linear)
    if 'gpr' in list_name:
        gpr = GaussianProcessRegressor(**choice_model_params['gpr'])
        base_models.append(gpr)
    return base_models

def single_model(choice_model_params, list_name):
    model_list = init_base_models(choice_model_params, list_name)
    for i,model in enumerate(model_list):
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        print('name:{0} ,dod:{1}'.format(list_name[i],dod(test_y, pred)))

def stacking(choice_model_params, list_name):
    base_models = init_base_models(choice_model_params, list_name)
    meta_model = RandomForestRegressor()    # LinearRegression()
    sm = StackingModels(base_models, meta_model, n_folds=10)
    sm.fit(train_X, train_y)
    linear_pred = sm.predict(test_X)
    return linear_pred


# list_name = ['svm','knn','xgboost','lightgbm','randomforest','linear']
list_name = ['knn','randomforest','linear','gpr']
file_path = '../data/df_real_measure_20190902_1.csv'
obtain_data(file_path)
spaces = create_space(list_name)
model_params = find_params(spaces,MAX_EVALS=1000)
choice_model_params = transform_choice(model_params, list_name)

# 单个模型预测
single_model(choice_model_params, list_name)

# Stacking集成预测
pred_result = stacking(choice_model_params, list_name)
print(pred_result)
print(dod(test_y, pred_result))