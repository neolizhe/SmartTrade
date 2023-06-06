# coding:utf-8
# Use stacked Conv2d/Pool2d layer instead of Wavelet decomposed layer
# In temporal direction --- Use multi Conv1d
# In spatial direction --- Use Cross layer
import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Concatenate, \
    Input, Conv2D, Lambda, GRU, Reshape, Dropout, BatchNormalization
import tensorflow as tf
from .CrossLayer import CrossLayer
from .LSTMmodel import LSTMmodel
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import LayerNormalization
from keras_layer_normalization import LayerNormalization


class ConvNet(LSTMmodel):
    def __init__(self, df, date_range, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        super().__init__(df, date_range, label_col,
                         window_size, test_size, shuffle, epochs, batch_size,
                         **kwargs)
        self.conv_order = 6

    def build_model(self, input_shape):
        input_vector = Input(shape=input_shape)
        ln_vector = LayerNormalization()(input_vector)
        conv_input = Reshape(target_shape=(input_shape[0], input_shape[1], 1))(ln_vector)
        # Construct multi conv1d filters
        conv_stack = []
        for index in range(self.conv_order):
            conv_out = Conv2D(filters=2, kernel_size=(2, 1), strides=(2, 1), name="conv2d_layer_%s" % index)(conv_input)
            origin_shape = conv_out.shape
            reshape_out = Reshape(target_shape=(origin_shape[1], origin_shape[2] * origin_shape[3], 1))(conv_out)

            flatten_conv = Conv2D(filters=1, kernel_size=(origin_shape[1], 1), strides=(1,1), name="extend_conv_%s" % index)(conv_out)
            conv_stack.append(flatten_conv)
            conv_input = reshape_out
        concat_out = Concatenate(axis=2)(conv_stack)
        print(concat_out.shape)
        # deep & cross layers
        reshape_deep = Reshape(target_shape=(concat_out.shape[-2],))(concat_out)
        # deep_input = reshape_deep
        # for i in range(4):
        #     deep_out = Dense(512, activation='relu')(deep_input)
        #     deep_drop = Dropout(0.2)(deep_out)
        #     deep_input = deep_drop
        dense1 = Dense(512, activation='relu')(reshape_deep)
        drop1 = Dropout(0.2)(dense1)
        dense2 = Dense(256, activation='relu')(drop1)
        dense3 = Dense(128, activation='relu')(dense2)
        dense4 = Dense(64, activation='relu')(dense3)
        final_out = Dense(1, activation='linear')(dense4)
        model = Model(input_vector, final_out)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        print(model.summary())
        self.model = model

    def fit_process(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=20,
                                                          restore_best_weights=True)
        history = self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                                 callbacks=[early_stopping])
        loss_hist = history.history['loss']
        self.loss = loss_hist


if __name__ == "__main__":
    import pandas as pds

    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    date_range = data.time.values
    data["label_copy"] = data.ths_close_price_stock
    data.drop(["Unnamed: 0", "time", "thscode"], axis=1, inplace=True)
    wave_lstm = ConvNet(df=data, date_range=date_range,
                             label_col='ths_close_price_stock',
                             window_size=128, epochs=200, batch_size=4, shuffle=False)
    wave_lstm.process()
