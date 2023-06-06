# coding:utf-8
# Use stacked Conv1d layer instead of Wavelet decomposed layer
# In temporal direction --- Use multi Conv1d
# In spatial direction --- Use Cross layer
import keras
from keras import Sequential, Model
from keras.layers import Dense, LSTM, Conv1D, Concatenate, \
    Input, Conv2D, Lambda, GRU, Reshape, Dropout, MaxPool2D
import tensorflow as tf
from .CrossLayer import CrossLayer
from .LSTMmodel import LSTMmodel
import matplotlib.pyplot as plt


class ConvPlusLSTM(LSTMmodel):
    def __init__(self, df, date_range, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        super().__init__(df, date_range, label_col,
                         window_size, test_size, shuffle, epochs, batch_size,
                         **kwargs)
        self.conv_order = 4

    def build_model(self, input_shape):
        input_vector = Input(shape=input_shape)
        conv_input = Reshape(target_shape=(input_shape[0], input_shape[1], 1))(input_vector)
        # Construct multi conv1d filters
        for index in range(self.conv_order):
            conv_out = Conv2D(filters=2, kernel_size=(2, 1), strides=(2, 1), name="conv2d_layer_%s" % index)(conv_input)
            pool_out = MaxPool2D(pool_size=(2, 1), strides=(2, 1), name="pool2d_layer_%s" % index)(conv_input)
            origin_shape = conv_out.shape
            reshape_out = Reshape(target_shape=(origin_shape[1], origin_shape[2] * origin_shape[3], 1))(conv_out)
            concat_out = Concatenate(axis=2)([reshape_out, pool_out])
            conv_input = concat_out

        last_conv_out = Conv2D(filters=1, kernel_size=(conv_input.shape[1], 1), name="conv2d_layer_output")(concat_out)
        final_reshape_out = Reshape(target_shape=(last_conv_out.shape[-2],))(last_conv_out)
        dropout = Dropout(rate=0.2)(final_reshape_out)
        dense1 = Dense(256, activation='relu')(dropout)
        # deep & cross layers
        reshape_cross = Reshape(target_shape=(1,-1))(dense1)
        cross_out = CrossLayer(output_dim=dense1.shape[-1], num_layer=4)(reshape_cross)
        deep_input = cross_out
        for i in range(4):
            deep_out = Dense(128,activation='relu')(deep_input)
            deep_drop = Dropout(0.4)(deep_out)
            deep_input = deep_drop

        concat_out = Concatenate(axis=-1)([cross_out, deep_drop])
        dense2 = Dense(64, activation='relu')(concat_out)
        final_out = Dense(1, activation='linear')(dense2)
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
    data["label_copy"] = data.ths_close_price_stock
    # data = data.iloc[:-200,:]
    date_range = data.time.values
    wave_lstm = ConvPlusLSTM(df=data, date_range=date_range,
                             label_col='ths_close_price_stock',
                             window_size=128, epochs=100, batch_size=4, shuffle=False)
    wave_lstm.data_process()
    wave_lstm.dataset_display()
    # wave_lstm.process()
