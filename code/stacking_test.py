import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, base_models, meta_model, n_folds=3):
		self.base_models = base_models
		self.meta_model = meta_model
		self.n_folds = n_folds

	def fit(self, X, y):
		# self.base_models_是用来保存每折的训练模型，predect时对测试集使用
		self.base_models_ = [list() for x in self.base_models]
		self.meta_model_ = clone(self.meta_model)
		kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=999)

		# 每个模型做cv验证，并得到每次的测试集预测结果（总的结果为整个数据集）
		cv_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
		for i, model in enumerate(self.base_models):
			for train_index, test_index in kfold.split(X, y):
				clf = clone(model)
				clf.fit(X[train_index], y[train_index])
				self.base_models_[i].append(clf)
				y_pred = clf.predict(X[test_index])
				cv_fold_predictions[test_index,i] = y_pred

		# 使用次级模型拟合次级训练集，不写在下面是因为fit的时间都计算在fit函数中
		self.meta_model_.fit(cv_fold_predictions, y)
		return self

	def predict(self, X):
		meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_])
		return self.meta_model_.predict(meta_features)



# # 使用简介
# from stacking_test import StackingModels

# rf = RandomForestRegressor()
# lr = ......
# base_models = [rf,lr,knn,...]
# # 选择二阶训练的模型，比如线性回归
# meta_model = LinearRegression()
# train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, test_size=0.25, random_state=999)


# # 传入融合模型列表、元模型、折数
# sm = StackingModels(base_models, meta_model, n_folds=5)
# # 将DataFrame转换为ndarray格式
# sm.fit(train_X.get_values(),train_y.get_values())
# tree_pred = sm.predict(test_X.get_values())