# coding : utf-8
from keras.layers import Lambda, Layer, Add, Flatten
from keras.backend import sum, batch_dot, reshape


class CrossLayer(Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, "Err input shape, must be (None,1,N) !"
        self.input_dim = input_shape[-1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape=[1, self.input_dim], initializer='glorot_uniform', name='w_' + str(i),
                                          trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i), trainable=True))
        self.built = True

    def call(self, inputs, **keywords):
        cross = inputs
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()(
                    [sum(self.W[i] * batch_dot(reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims=True),
                     self.bias[i], x]))(inputs)
            else:
                cross = Lambda(lambda x: Add()(
                    [sum(self.W[i] * batch_dot(reshape(x, (-1, self.input_dim, 1)), inputs), 1, keepdims=True),
                     self.bias[i], inputs]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return None, self.output_dim
