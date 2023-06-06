# coding : utf - 8
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import *

@registry.register_model
class TemporalTransformer(Transformer):
    def __init__(self,*args,**kwargs):
        super(TemporalTransformer, self).__init__(**args,**kwargs)
        self._hparams = transformer_base_v1()


    def body(self, features):
        hparams = self._hparams
        fshape = common_layers.shape_list(features)
        decoder_input,label = tf.split(features,[fshape[-1]-1, 1],axis=-1)
        label = features["label"]
        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(
                common_layers.shape_list(label)[1]))
        decoder_output = self.transformer_decoder(
            decoder_input,
            decoder_self_attention_bias
        )
        return tf.reshape(decoder_output, common_layers.shape_list(label))


    def wavelet_embedding(self, target):
        with tf.variable_scope("wavelet_embed"):
        # return embedding_tensor

    def transformer_decoder(self,
                            decoder_input,
                            decoder_self_attention_bias):
        # A stack of transformer layers
        # Each layers contains masked multi head attention and
        # feedforward sublayer and skip-connection
        x = decoder_input
        # Step 1: Add casual bias to tensor(Masked attention)
        # masked attention limited to each position's left
        # A bias tensor shared the same shap as decoder input
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
            hparams=self._hparams,
            encoder_output=None,
            encoder_decoder_attention_bias=None
        )
        return x