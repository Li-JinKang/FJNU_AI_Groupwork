from sklearn.model_selection import train_test_split
from dataCleaned import *
import resultShow
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

dataset = cleanedData(file_path='../data.csv')
data, labels, target = dataset.dataRead(['id', 'target'])
# labels, target = dataset.dataClean(['Sex'], [[0, 1], ], ['id', 'target'])

# 划分数据集
# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20, random_state=42)

# 创建LGBMClassifier实例，并设置class_weight参数
# Create LGBMClassifier instance and set class_weight parameter
# clf = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
# clf = lgb.LGBMClassifier(random_state=42, boosting_type='rf')

# 创建随机森林分类器
# Create a RandomForestClassifier
rf = RandomForestClassifier()

# 训练模型
# Train the model
rf.fit(x_train, y_train)

# 展示得分
# Show the evaluation scores
score = resultShow.scoreShow(rf, x_train, x_test, y_train, y_test).show()


# 以下是未启用的随机森林调优代码：
# The following is the commented-out Random Forest hyperparameter tuning code:

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
# from sklearn.preprocessing import StandardScaler
#
# from dataCleaned import cleanedData
#
# dataset = cleanedData(file_path='data.csv')
# data, labels, target = dataset.dataRead(['id', 'target'])
#
# # 数据预处理
# # Data preprocessing
# # scaler = StandardScaler()
# # x_scaled = scaler.fit_transform(labels)  # Standardize features
#
# # 划分数据集
# # Split the dataset into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.2, random_state=42)
#
# # 使用 SMOTE 进行过采样
# # Use SMOTE for oversampling
# # smote = SMOTE(random_state=42)
# # x_train, y_train = smote.fit_resample(x_train, y_train)
#
# # 使用随机森林模型并调优
# # Use Random Forest model with hyperparameter tuning
# rf = RandomForestClassifier(random_state=42, class_weight='balanced')  # Use class_weight to handle class imbalance
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5],
# }
# grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted')
# grid_search.fit(x_train, y_train)
#
# # 最优模型
# # Best model
# best_rf = grid_search.best_estimator_
#
# # 评估模型
# # Evaluate the model
# y_pred = best_rf.predict(x_test)
# print("Random Forest Model Evaluation:")
# print(classification_report(y_test, y_pred))
# print('f1_score:', f1_score(y_test, y_pred, average='macro'))
