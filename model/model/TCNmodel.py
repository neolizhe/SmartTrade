# coding : utf - 8
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import pandas as pds
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Concatenate, \
    Input, Conv2D, Lambda, GRU, Reshape, Dropout, BatchNormalization, Add
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from model.model.ConvNet import ConvNet


class WeightNormCasualConvLayer(tf.keras.layers.Conv1D):
    def __init__(self, *args, **kwargs):
        self.weight_norm = kwargs.pop("weight_norm")
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        # weight normalization
        if self.weight_norm:
            self.scalar_g = self.add_weight(
                name="scalar_g",
                shape=(self.filters,),
                initializer=tf.random_normal_initializer(),
                trainable=True,
                dtype=self.kernel.dtype
            )
            self.kernel = tf.reshape(self.scalar_g, [1, 1, self.filters]) * tf.math.l2_normalize(self.kernel, axis=[0, 1])

        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()

        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self.built = True


class TCNmodel(ConvNet):
    def __init__(self, df: pds.DataFrame, date_range: list, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        super().__init__(df, date_range, label_col,
                         window_size, test_size, shuffle, epochs, batch_size,
                         **kwargs)
        if "conv_size" in kwargs.keys():
            self.conv_window_size = kwargs["conv_size"]
        else:
            self.conv_window_size = window_size
        self.filters = 32
        self.dilation_rate = 2

    def residual_block(self, filters, dilation_rate, input_vector):
        # first layer of weighted norm casual conv1d
        conv_vector = WeightNormCasualConvLayer(
            filters=filters,
            kernel_size=self.conv_window_size,
            padding="causal",
            dilation_rate=dilation_rate,
            weight_norm=True,
            activation="relu",
            dtype=tf.float32
        )(input_vector)
        # Dropout
        dropout_vector = Dropout(
            rate=0.2
        )(conv_vector)
        # second layer of weighted norm conv
        conv_vector_2 = WeightNormCasualConvLayer(
            filters=filters,
            kernel_size=self.conv_window_size,
            padding="causal",
            dilation_rate=dilation_rate,
            weight_norm=True,
            activation="relu",
            dtype=tf.float32
        )(dropout_vector)
        dropout_vector_2 = Dropout(rate=0.2)(conv_vector_2)
        # Residual component
        residual_vector = input_vector
        if input_vector.shape[-1] != filters:
            residual_vector = Conv1D(
                filters=filters,
                kernel_size=1
            )(input_vector)
        print("input shape:%s, dropout shape:%s, residual shape:%s" % (input_vector.shape, dropout_vector_2.shape, residual_vector.shape))
        add_vector = Add()([residual_vector , dropout_vector_2])
        inline_func = lambda x : tf.keras.activations.relu(x)
        residual_output_vector = Lambda(inline_func)(add_vector)
        return residual_output_vector

    def build_model(self, input_shape):
        input_vector = Input(shape=input_shape)
        mid_vector = input_vector
        dilation_rate, filters = self.dilation_rate, self.filters
        for index in range(4):
            print(index, filters, dilation_rate)
            conv_vector = self.residual_block(
                filters=filters,
                dilation_rate=dilation_rate,
                input_vector=mid_vector
            )
            filters = filters // 2
            dilation_rate *= 2
            mid_vector = conv_vector
        # Dense output
        conv_shape = conv_vector.shape
        reshape_vector = Reshape(target_shape=(conv_shape[-2] * conv_shape[-1],))(conv_vector)
        dense1 = Dense(128, activation='relu')(reshape_vector)
        dense2 = Dense(64, activation='relu')(dense1)
        final_out = Dense(1, activation='sigmoid')(dense2)
        model = Model(input_vector, final_out)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        print(model.summary())
        self.model = model


if __name__ == "__main__":

    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    date_range = data.time.values
    data["label_copy"] = data.ths_close_price_stock
    data.drop(["Unnamed: 0", "time", "thscode"], axis=1, inplace=True)
    tcn = TCNmodel(df=data, date_range=date_range,
                             label_col='ths_close_price_stock',
                             window_size=128, epochs=50, batch_size=4, shuffle=False)
    tcn.process()
