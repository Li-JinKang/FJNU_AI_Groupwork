from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score

class scoreShow():
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        # 训练数据和测试数据
        # Training and testing data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # 模型的预测值
        # Model predictions
        self.y_pred = self.model.predict(self.x_test)

    def show(self):
        train_score = self.model.score(self.x_train, self.y_train)
        test_score = self.model.score(self.x_test, self.y_test)
        print(f'train_score:{train_score}\ntest_score:{test_score}')

        # 准确率
        # Accuracy
        accuracy_score_value = accuracy_score(self.y_test, self.y_pred)
        # 召回率
        # Recall
        recall_score_value = recall_score(self.y_test, self.y_pred, average='macro')
        # 精确率
        # Precision
        precision_score_value = precision_score(self.y_test, self.y_pred, average='macro')

        classification_report_value = classification_report(self.y_test, self.y_pred)

        print(
            f'准确率:{accuracy_score_value}\n召回率:{recall_score_value}\n精确率:{precision_score_value}\n分类报告:\n{classification_report_value}')

        print('f1_score(macro):', f1_score(self.y_test, self.y_pred, average='macro'))
