# coding:utf-8
import json
import math
import numpy as np
import pandas as pds
from datetime import datetime, timedelta
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from config.rootPath import root


def sigmoid(x, lb, ub):
    # lb -> output 0.05
    # ub -> output 0.95
    if lb == ub:
        return 0
    b = np.log((1 / 0.05 - 1) / (1 / 0.95 - 1)) / (ub - lb)
    a = np.log(1 / 0.05 - 1) + b * lb
    return 1 / (1 + math.exp(a - b * x))


class LSTMmodel:
    def __init__(self, df: pds.DataFrame, date_range: list, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        self.data = df.reset_index(drop=True)
        self.test_size = test_size
        self.label_col = label_col
        self.window_size = window_size
        self.params = {"lstm_layers": 1, "lstm_units": [128, ]}
        self.encoding_map = {}
        self.scale_map = {label_col: 1 / 1000, }
        self.date_range = date_range
        self.shuffle = shuffle
        # inner model params
        self.epochs = epochs
        self.batch_size = batch_size

        # process variables
        self.model = Sequential()
        self.X_train = []
        self.Y_train = []
        self.x_test = []
        self.y_test = []
        self.loss = []

        for key in kwargs.keys():
            self.params[key] = kwargs.get(key)

    def data_process(self):
        test_lens = int(self.test_size * len(self.data))
        # pre process for total data
        for col in self.data.columns:
            if (self.data[col].isnull().all()) or (len(self.data[col].dropna().unique()) < 2):
                self.data.drop(col, axis=1, inplace=True)
            elif self.data[col].isnull().any():
                origin_value = self.data[col].dropna().values[0]
                col_values = self.data[col].values
                filled_col = []
                for i in range(len(col_values)):
                    if pds.isnull(col_values[i]):
                        filled_col.append(origin_value)
                    else:
                        filled_col.append(col_values[i])
                        origin_value = col_values[i]
                self.data[col] = filled_col
            else:
                continue

        self.test_data = self.data.iloc[-test_lens:, :].copy()
        self.train_data = self.data.iloc[:-test_lens, :].copy()

        # preprocess data set
        cols = self.train_data.columns
        for col in cols:
            if self.train_data[col].dtype in (str, object):
                enums = self.train_data[col].dropna().unique()
                enums_dict = {}
                for i in range(len(enums)):
                    enums_dict[enums[i]] = i
                self.train_data[col] = self.train_data[col].map(
                    lambda x: int(len(enums) / 2) if pds.isnull(x) else enums_dict[x])
                self.test_data[col] = self.test_data[col]. \
                    map(lambda x: int(len(enums) / 2) if pds.isnull(x) or x not in enums_dict.keys() else enums_dict[x])
                self.encoding_map[col] = enums_dict

            self.train_data[col] = self.train_data[col].astype(float)
            self.test_data[col] = self.test_data[col].astype(float)

            fillna_value = self.train_data[col].mean()
            self.train_data[col] = self.train_data[col].fillna(fillna_value)
            self.test_data[col] = self.test_data[col].fillna(fillna_value)

        # scale dataset
        for col in cols:
            lb, ub = 0, 1
            if len(self.train_data[col].unique()) > 100:
                lb, ub = self.train_data[col].quantile(0.02), self.train_data[col].quantile(0.98)
            else:
                lb, ub = self.train_data[col].min(), self.train_data[col].max()

            if col == self.label_col:
                self.train_data[col] = self.train_data[col] * self.scale_map[col]
                self.test_data[col] = self.test_data[col] * self.scale_map[col]
            else:
                self.train_data[col] = self.train_data[col].apply(lambda x: sigmoid(x, lb, ub))
                self.test_data[col] = self.test_data[col].apply(lambda x: sigmoid(x, lb, ub))

        # ensamble train data (N, timestep, features)
        train_sample = []
        features = self.train_data.drop(self.label_col, axis=1).values
        for index in range(len(features) - self.window_size):
            time_stamps = []
            for i in range(index, index + self.window_size):
                time_stamps.append(features[i])
            train_sample.append(time_stamps)

        X_train = np.array(train_sample)
        Y_train = self.train_data[self.label_col].values[self.window_size:]
        if self.shuffle:
            shuffled_index = np.arange(len(X_train), dtype=int)
            np.random.shuffle(shuffled_index)
            X_train = X_train[shuffled_index]
            Y_train = Y_train[shuffled_index]
        # random.rand
        # generate test set input
        test_sample = []
        features_test = features[-self.window_size:]
        features_test = np.append(features_test, self.test_data.drop(self.label_col, axis=1).values, axis=0)
        for index in range(len(features_test) - self.window_size + 1):
            time_stamps = []
            for i in range(index, index + self.window_size):
                time_stamps.append(features_test[i])
            test_sample.append(time_stamps)

        x_test = np.array(test_sample)
        y_test = self.test_data[self.label_col].values
        y_test = np.append(y_test, np.array([0]), axis=0)

        self.X_train, self.Y_train, self.x_test, self.y_test = X_train, Y_train, x_test, y_test

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=128,
                       input_shape=input_shape,
                       # activation='relu',
                       return_sequences=True))

        # for i in range(self.params['lstm_layers'] - 1):
        model.add(LSTM(units=64,
                       # activation='relu',
                       return_sequences=False))

        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        print(model.summary())
        self.model = model

    def fit_process(self):
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict_process(self):
        x_test = np.append(self.X_train[-len(self.x_test):], self.x_test, axis=0)
        y_test = np.append(self.Y_train[-len(self.y_test):], self.y_test, axis=0)
        y_pred = self.model.predict(x_test)
        print("test set MSE:%s, MAPE:%s" % (mean_squared_error(y_test[:-1], y_pred[:-1]),
                                            mean_absolute_percentage_error(y_test[:-1], y_pred[:-1])))
        dates = self.date_range[-len(y_test) + 1:]
        # get last date
        last_time = datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=1)
        last_date = datetime.strftime(datetime(last_time.year, last_time.month, last_time.day), "%Y-%m-%d")
        dates = np.append(dates, last_date)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(dates, y_pred, label="Prediction")
        plt.plot(dates[:-1], y_test[:-1], label="True value")
        plt.legend()
        plt.grid(axis='x')
        plt.xticks(rotation=90, fontsize=4)
        plt.xlabel("Date", fontsize=8)
        plt.ylabel("Close stock price /DAY", fontsize=8)
        plt.axvline(x=int(len(dates) / 2), ls="-", c="green")
        # plt.xlim(xmin="2022-11-01")

        if self.loss:
            plt.subplot(2, 1, 2)
            plt.plot(self.loss)
            plt.yscale('log')

        plt.show()

    def process(self):
        self.data_process()
        self.build_model(input_shape=self.X_train.shape[1:])
        self.fit_process()
        self.predict_process()

    def dataset_display(self):
        for col in self.train_data.columns:
            print(col, self.train_data[col].max())


if __name__ == "__main__":
    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    # data = data.iloc[:-200,:]
    date_range = data.time.values
    data.drop(["Unnamed: 0", "time", "thscode"], axis=1, inplace=True)
    dicts = {}
    for col in data.columns:
        tmp = pds.DataFrame({"a": data.ths_close_price_stock.values,
                             "b": data[col].values})
        v = tmp.corr().values[0]
        if len(v) > 1:
            dicts[col] = v[1]
    b = sorted(dicts.items(), key=lambda x: x[1])
    json_path = root("StockInfoBook.json")
    with open(json_path, "r", encoding="UTF-8") as js:
        stock_dict = json.load(js)
    stock_info_dict = {}
    for k, v in stock_dict.items():
        if isinstance(v, list):
            stock_info_dict[v[0]] = k
        else:
            stock_info_dict[v] = k
    for k, v in b:
        if k in stock_info_dict.keys():
            print(stock_info_dict[k], "\t", v)
        else:
            print(k, "\t", v)
    # data.drop(['ths_open_price_stock', 'ths_low_stock', 'ths_high_price_stock'],
    #           axis=1, inplace=True)
    lstm = LSTMmodel(df=data, date_range=date_range,
                     label_col='ths_close_price_stock',
                     window_size=128, epochs=100, batch_size=4, shuffle=False)
    lstm.fit_process()
