# coding : utf - 8
'''
Novel model combine transformer and temporal encoding
'''
import numpy as np
from keras import Sequential,Model
from keras.layers import Dense, LSTM, Conv1D, Concatenate, Input, Conv2D, Lambda, GRU
import tensorflow as tf
from tensor2tensor.layers.common_attention import *
from tensor2tensor.layers.common_layers import *
from tensor2tensor.models.transformer import *
from tensor2tensor.layers.common_hparams import *
from tensor2tensor.utils.trainer_lib import *
from model.model.LSTMmodel import LSTMmodel

class TemporalTransformer(LSTMmodel):
    def __init__(self, input_tensor):
        super(TemporalTransformer, self).__init__()
        self.stack_layers = 6
        self.input_tensor = input_tensor
        self.hparams = common_hparams.basic_params1()
        self.hparams_modification()

    def hparams_modification(self):
        self.hparams.set_hparam("batch_size", 128)

    def preprocess_input_tensor(self, input_tensor):
        # transform temporal sequence to embedding vector
        # within a specific window
        target, _ = tf.split(input_tensor,
                          [1, input_tensor.shape[-1] - 1],
                          axis=-1)
        embedding_tensor = self.wavelet_embedding(target)
        processed_input_tensor = tf.concat([embedding_tensor])
        return processed_input_tensor

    def wavelet_embedding(self, target):
        with tf.variable_scope("wavelet_embed"):

        # return embedding_tensor

    def build_model(self, input_shape):
        #Input tensor
        input_tensor = Input(shape=input_shape)
        # encoding
        decoder_input = self.preprocess_input_tensor(input_tensor)
        # decoder - composed of casual or masked self attention
        # and skip connection
        output_tensor = self.transformer_decoder(decoder_input)
        model = Model(input_tensor, output_tensor)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        print(model.summary())
        self.model = model

    def transformer_decoder(self, decoder_input):
        # A stack of transformer layers
        # Each layers contains masked multi head attention and
        # feedforward sublayer and skip-connection
        x = decoder_input
        # Step 1: Add casual bias to tensor(Masked attention)
        # masked attention limited to each position's left
        # A bias tensor shared the same shap as decoder input
        decoder_self_attention_bias = (
            attention_bias_lower_triangle(shape_list(x)[1]))

        with tf.variable_scope("decoder"):
            for index in range(self.stack_layers):
                x = self.single_decoder_layer(
                    x,
                    decoder_self_attention_bias,
                    index
                )
            return common_layers.layer_preprocess(x, self.hparams)

    def single_decoder_layer(self,
                             decoder_input,
                             bias,
                             index
                             ):
        x = transformer_decoder_layer(
            decoder_input,
            decoder_self_attention_bias=bias,
            layer_idx=index,
            hparams=self.hparams,
            encoder_output=None,
            encoder_decoder_attention_bias=None
        )
        return x




