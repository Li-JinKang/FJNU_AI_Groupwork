from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.over_sampling import SMOTE
import pandas as pd

data = pd.read_csv("data.csv")
labels = data.iloc[:, 1:-1]
target = data.iloc[:, -1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20)

smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)

model = KNN(n_neighbors=2, metric='manhattan')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
classification_report_value = classification_report(y_test, y_pred)
print(f'分类报告:\n{classification_report_value}')
print('f1_score(macro):', f1_score(y_test, y_pred, average='macro'))