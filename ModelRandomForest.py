from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_csv("data.csv")
labels = data.iloc[:, 1:-1]
target = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20, random_state=42)

rf = RandomForestClassifier()

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

classification_report_value = classification_report(y_test, y_pred)
print(f'分类报告:\n{classification_report_value}')
print('f1_score(macro):', f1_score(y_test, y_pred, average='macro'))