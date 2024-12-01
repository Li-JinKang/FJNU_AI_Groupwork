import copy

import pandas as pd

class cleanedData():
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1]
        # self.data 完整数据; self.label 带有标签的数据; self.target 数据的最终值
        self.data, self.labels, self.target = None, None, None

    # 对数据进行基本处理
    def dataRead(self, columns_exclude: list):
        if self.file_type.lower() == 'csv':
            # 读取数据
            self.data = pd.read_csv(self.file_path)
            # 数据切片（划分为标签和目标，同时去掉索引列）
            self.target = self.data.iloc[:, -1]
            self.labels = copy.deepcopy(self.data)
            self.labels = self.labels.drop(columns=columns_exclude)
        return self.data, self.labels, self.target


    def dataClean(self, columns_clean: list, corresponding_args: list, columns_exclude, type=1):
        self.dataRead(columns_exclude)
        if type:
            data = copy.deepcopy(self.data)
            # 删除数据有误的行
            for index, column in enumerate(columns_clean):
                data = data[~((data[column] > corresponding_args[index][0]) & (data[column] < corresponding_args[index][1]))]

            target = data.iloc[:, -1]
            labels = copy.deepcopy(data)
            labels = labels.drop(columns=columns_exclude)
            return labels, target


if __name__ == '__main__':
    pass
    # temp = cleanedData(file_path='data.csv')
    # data, labels, target = temp.dataRead(['id', 'target'])
    # print(data)
    # print(labels)
    # print(target)
    # print(temp.dataClean(['Sex'], [[0, 1], ], ['id', 'target']))
    data = pd.read_csv("data.csv")
    for label in data.keys():
        target_counts = data[label].value_counts()
        print(target_counts)