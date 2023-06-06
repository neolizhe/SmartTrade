# coding:utf-8
import numpy as np
from keras import Sequential,Model
from keras.layers import Dense, LSTM, Conv1D, Concatenate, Input, Conv2D, Lambda, GRU
import tensorflow as tf

from model.model.LSTMmodel import LSTMmodel


class WaveletPlusLSTM(LSTMmodel):
    def __init__(self, df, date_range, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        super().__init__(df, date_range, label_col,
                         window_size, test_size, shuffle, epochs, batch_size,
                         **kwargs)
        self.wavelet_order = 4

    def build_model(self, input_shape):
        # input definition
        input_vector = Input(shape=input_shape)
        wavelet_vector = input_vector
        wavelet_weight = np.sqrt(2) / 2  # wavelet decompose coef
        detail_layers = []
        for index in range(self.wavelet_order):
            input_vectors = Lambda(tf.split, arguments={'num_or_size_splits':wavelet_vector.shape[1] // 2, 'axis':1},
                                   name="approx_layer_%s" % index)(wavelet_vector)
            print(input_vector.shape, index)
            approx_op = wavelet_weight * tf.ones(shape=(1,2))
            detail_op = wavelet_weight * tf.constant([1.0, -1.0], shape=(1,2))
            approx_list, detail_list = [],[]
            j = 0
            for inputs in input_vectors:
                approxis = Lambda(lambda x:tf.matmul(approx_op,x),name="approx_op_%s_%s" % (index, j))(inputs)
                details = Lambda(lambda x:tf.matmul(detail_op,x),name="detail_op_%s_%s" % (index, j))(inputs)
                approx_list.append(approxis)
                detail_list.append(details)
                j += 1
            approx_vector = Concatenate(axis=1)(approx_list)
            detail_vector = Concatenate(axis=1)(detail_list)
            if index == self.wavelet_order - 1:
                detail_vector = Concatenate(axis=-1)([detail_vector, approx_vector])
            detail_layers.append(detail_vector)
            wavelet_vector = approx_vector

        output_vector_list = []
        for vector in detail_layers:
            output_vector = LSTM(units=128,dropout=0.2,
                                 # input_shape=vector.shape,
                                 # activation='relu',
                                 return_sequences=False)(vector)
            output_vector_list.append(output_vector)

        # concatenate output vector
        concate_out = Concatenate(axis=-1)(output_vector_list)
        dense_out = Dense(units=128,activation="relu")(concate_out)
        final_out = Dense(units=1,activation="linear")(dense_out)
        print(input_vector.shape, final_out.shape)
        model = Model(input_vector, final_out)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        print(model.summary())
        self.model = model

if __name__ == "__main__":
    import pandas as pds
    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    data["label_copy"] = data.ths_close_price_stock
    # data = data.iloc[:-200,:]
    date_range = data.time.values
    wave_lstm = WaveletPlusLSTM(df=data, date_range=date_range,
                     label_col='ths_close_price_stock',
                     window_size=128, epochs=100, batch_size=4, shuffle=False)
    wave_lstm.process()

