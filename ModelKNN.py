from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.over_sampling import SMOTE
import pandas as pd

data = pd.read_csv("data.csv")
labels = data.iloc[:, 1:-1]
target = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20)

# 使训练数据均衡
# Balance the training data.
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)

# 参数为经过GridSearchCV优化的最佳结果
# The parameters are the best results obtained after optimizing with GridSearchCV.
model = KNN(n_neighbors=2, metric='manhattan')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
classification_report_value = classification_report(y_test, y_pred)
print(f'分类报告:\n{classification_report_value}')
print('f1_score(macro):', f1_score(y_test, y_pred, average='macro'))