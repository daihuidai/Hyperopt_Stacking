import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_style("whitegrid", {"font.sans-serif":['KaiTi', 'Arial']})
mpl.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_rows', 5000)


df = pd.read_csv('../data/measure_real.csv', index_col=0)
# df = df[(df['source'] == 'picc') & (df['weight'] < 100)]
columns = df.columns[2:]
# columns = ['weight', 'paper_l', 'paper_w', 'paper_a', 'paper_angle', 'paper_dip', 'body_l', 'body_l_j', 'body_h', 'body_h_j', 'forebody_w',
#  'hindbody_w', 'heart_girth', 'heart_girth_j', 'waist', 'legs_d', 'legs_d_j', 'chest_waist_d', 'chest_waist_d_j', 'trunk_a', 'trunk_legs_a',
#   'chest_waist_a', 'hexagon_a', 'hexagon_head_a', 'hexagon_chest_a', 'hexagon_back_a', 'hexagon_ass_a', 'nose_chest_dr', 'chest_waist_dr']

# fig, ax = plt.subplots()
df_corr = df.corr(method='pearson')
# sns.heatmap(df_corr, annot=True, ax=ax, annot_kws={'size':8}, yticklabels=True, xticklabels=True, cmap='Greys')
# ax.set_xticklabels(df_corr.index, rotation=45)
# ax.set_yticklabels(df_corr.index, rotation=0)
# plt.show()

# pearson_corr = df_corr.iloc[0].sort_values(ascending=False)
# rcolumns = ['weight', 'heart_girth_j', 'heart_girth', 'hexagon_a', 'trunk_a', 'waist', 'trunk_legs_a', 'hexagon_back_a', 'body_l_j', 'hexagon_chest_a',
#  'chest_waist_a', 'hexagon_head_a', 'body_l', 'body_h_j', 'body_h', 'forebody_w', 'hexagon_ass_a', 'legs_d', 'legs_d_j', 'chest_waist_d_j', 'chest_waist_d',
#   'hindbody_w', 'nose_chest_dr']


df = df[columns]
df.drop('refer_type', axis=1, inplace=True)
df_X = df.iloc[:,1:]
df_y = df.iloc[:,0]
scaler = StandardScaler()
df_scaler = scaler.fit_transform(df_X)

# 测试贝叶斯优化
from hyperopt import hp
from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train_X, test_X, train_y, test_y = train_test_split(df_scaler, df_y, test_size=0.3, random_state=999)

model = linear_model.LinearRegression()
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(mean_squared_error(test_y, pred))


def objective(params, n_folds=10):
    model = linear_model.LinearRegression()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    mse = mean_squared_error(test_y, pred)
    return {'loss':mse, 'params':params, 'status': STATUS_OK}

space = {
    'fit_intercept': hp.choice('fit_intercept', [False, True]),
    'normalize': hp.choice('normalize', [False, True])
}

from hyperopt import tpe
tpe_algorithm = tpe.suggest

from hyperopt import Trials
bayes_trials = Trials()

from hyperopt import fmin
MAX_EVALS = 500
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=bayes_trials)
print(best)