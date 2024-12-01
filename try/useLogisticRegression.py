import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import dataCleaned
import resultShow
import numpy as np

# 读取数据
# Load the dataset
dataset = dataCleaned.cleanedData(file_path='../data.csv')
# data, labels, target = dataset.dataRead(['id', 'target'])
labels, target = dataset.dataClean(['Sex'], [[0, 1], ], ['id', 'target'])
# print(labels, target)

# 划分数据集
# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(labels, target, test_size=0.20)

# 忽略警告信息
# Ignore warning messages
# warnings.filterwarnings("ignore")

# 创建逻辑回归模型
# Create the Logistic Regression model
# model = LogisticRegression(multi_class='multinomial', max_iter=200, C=0.3, random_state=4)
model = LogisticRegression(multi_class='ovr', solver='lbfgs')

# 定义每个类别的权重
# Define the class weights for each class
weight_for_class_0 = 1
weight_for_class_1 = 15
weight_for_class_2 = 5

# 使用 numpy.select 来根据条件分配权重
# Use numpy.select to assign weights based on conditions
conditions = [
    y_train == 0,
    y_train == 1,
    y_train == 2
]
choices = [
    weight_for_class_0,
    weight_for_class_1,
    weight_for_class_2
]

# 计算样本权重
# Compute sample weights
sample_weight = np.select(conditions, choices)

# 输入训练数据
# Fit the model with training data and sample weights
model.fit(x_train, y_train, sample_weight=sample_weight)
# model.fit(x_train, y_train)

# 展示得分
# Show the evaluation scores
score = resultShow.scoreShow(model, x_train, x_test, y_train, y_test).show()
