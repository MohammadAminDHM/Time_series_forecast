import numpy as np
from pandas import read_csv, DataFrame
from sklearn import preprocessing

class Prepare_Data:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, sequence_number, feature_number, test_number):
        data = read_csv(filename)
        self.data = data.iloc[:, 1:]
        self.tes_len = test_number
        self.seq_len = sequence_number  # Number of each sequences data
        self.fea_len = feature_number  # Number of column data

    def Preprossesing(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        fit_transform_data = min_max_scaler.fit_transform(self.data)
        preprocessing_data = DataFrame(fit_transform_data, columns=self.data.columns)
        return preprocessing_data, min_max_scaler

    def get_train_test_data(self):
        data, transform = self.Preprossesing()
        data_stock = data.values

        result = []
        for index in range(len(data_stock) - (self.seq_len + 1)):
            result.append(data_stock[index: index + (self.seq_len + 1)])
        result = np.array(result)

        train_data = result[:-self.tes_len, :]
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1][:, -1]
        x_test = result[-self.tes_len:, :-1]
        y_test = result[-self.tes_len:, -1][:, -1]
        X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], self.fea_len))
        X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], self.fea_len))
        return X_train, y_train, X_test, y_test, transform





