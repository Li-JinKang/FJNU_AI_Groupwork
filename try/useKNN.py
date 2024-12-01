from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler

import dataCleaned
import resultShow
from imblearn.over_sampling import SMOTE

dataset = dataCleaned.cleanedData(file_path='../data.csv')
data, labels, target = dataset.dataRead(['id', 'target'])
# labels, target = dataset.dataClean(['Sex'], [[0, 1], ], ['id', 'target'])

# 划分数据集
# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20, random_state=42)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 使用SMOTE平衡数据
# Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 训练KNN模型
# Train the KNN model
model = KNN(n_neighbors=2, metric='manhattan')
model.fit(x_train, y_train)

# 评估模型
# Evaluate the model
score = resultShow.scoreShow(model, x_train, x_test, y_train, y_test).show()

# 以下是未启用的KNN模型优化代码：
# The following is the commented-out KNN model optimization code:

# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, roc_auc_score, f1_score
#
# import dataCleaned
#
# dataset = dataCleaned.cleanedData(file_path='data.csv')
# data, labels, target = dataset.dataRead(['id', 'target'])
# # labels, target = dataset.dataClean(['Sex'], [[0, 1], ], ['id', 'target'])
#
# # labels = labels.drop(['Age', 'Sex', 'CholCheck', 'AnyHealthcare'], axis=1)
#
# # # 数据预处理
# # Standardization
# # scaler = StandardScaler()
# # x_scaled = scaler.fit_transform(labels)
#
# # 划分数据集
# # Split the dataset into training and test sets
# # x_train, x_test, y_train, y_test = train_test_split(x_scaled, target, test_size=0.20, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20, random_state=42)
#
# # 使用SMOTE平衡数据
# # Use SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# x_train, y_train = smote.fit_resample(x_train, y_train)
#
# # KNN模型优化
# # KNN model optimization
# knn = KNeighborsClassifier()
# param_grid = {
#     'n_neighbors': [3, 5, 10],
#     'n_neighbors': [2, 3, 4],
#     'metric': ['manhattan', 'minkowski', 'euclidean']
#     'metric': ['manhattan']
# }
# grid_search_knn = GridSearchCV(knn, param_grid, cv=3)
# grid_search_knn.fit(x_train, y_train)
# best_knn = grid_search_knn.best_estimator_
#
# # 评估模型
# # Evaluate the model
# print(grid_search_knn.best_params_)
# y_pred = best_knn.predict(x_test)
# print("KNN Model Evaluation:")
# print(classification_report(y_test, y_pred))
# print('f1_score:', f1_score(y_test, y_pred, average='macro'))
