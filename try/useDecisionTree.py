import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import dataCleaned
import resultShow
import pandas as pd

dataset = dataCleaned.cleanedData(file_path='../data.csv')
data, labels, target = dataset.dataRead(['id', 'target'])
# labels = labels.loc[:,['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'DiffWalk', 'Education', 'PhysHlth']]
# labels = labels.loc[:, ['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'DiffWalk', 'Education', 'PhysHlth']]
# Select only relevant columns for the labels
labels = labels.drop(['Age', 'Sex', 'CholCheck', 'AnyHealthcare'], axis=1)
# labels, target = dataset.dataClean(['Sex'], [[0, 1], ], ['id', 'target'])

# 标准化
# Standardization
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(labels)

# 划分数据集
# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x_scaled, target, test_size=0.20, random_state=42)

# # 使用SMOTE平衡数据
# Use SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# 训练加权决策树
# Train a weighted decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'max_depth': [10, 15, 20],  # 最大深度
#     'min_samples_leaf': [2, 4, 6],  # 叶节点的最小样本数
#     'min_samples_split': [2, 10, 20],
#     'max_features': ['sqrt', 'log2', None],  # 分割特征最大值
# }
#
# 调整类别权重
# Adjust class weights
# class_weights = {0: 1.0, 1: 15.0, 2: 5.0}
# # cv 交叉验证次数
# grid_search = GridSearchCV(DecisionTreeClassifier(class_weight=class_weights, random_state=42),
#                            param_grid, cv=5, n_jobs=-1)
#
# grid_search.fit(x_train, y_train)
#
# best_model = grid_search.best_estimator_
#
# # 输出最佳参数组合
# # Output best parameter combination
# print("Best parameters:", grid_search.best_params_)

# 预测并评估模型
# Predict and evaluate the model
score = resultShow.scoreShow(clf, x_train, x_test, y_train, y_test).show()
# score = resultShow.scoreShow(best_model, x_train, x_test, y_train, y_test).show()
