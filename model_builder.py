"""
Hybrid 1D-CNN + Transformer for ECG Arrhythmia Classification — Model V-Binary

Extracted from `model Binary3.ipynb` so it can be imported as a module
(`import model_builder`) by the training pipeline notebook, exactly like
`train_with_noise.py` imports its own model definition.

This defines a binary classifier (Normal vs. Abnormal) built from:
  1) a 1D-CNN feature extractor,
  2) a relative positional encoding block,
  3) a stack of Transformer encoder blocks (custom, QAT-compatible),
  4) a classification head.

The `apply_qat` flag threaded through every builder function controls
whether custom layers are wrapped with `tfmot.quantization.keras.quantize_annotate_layer`
so that `tfmot.quantization.keras.quantize_apply` can later insert fake-quantization
nodes for Quantization-Aware Training.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

import tf_keras as keras
from tf_keras import layers, models
from tf_keras.layers import Dense, Layer
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
import tensorflow_model_optimization as tfmot

# ── Data shape (must match dataprep.ipynb) ─────────────────────────────
WINDOW_SIZE = 1080     # samples per window: 3 sec @ 360 Hz
NUM_CLASSES = 2         # Binary: Normal vs. Abnormal
CLASS_NAMES = ["Normal", "Abnormal"]

# ── 1D-CNN front-end ──────────────────────────────────────────────────────
# First stride is 3, reducing the sequence length faster than a stride of 2
# would, which keeps Transformer attention cost down.
# Sequence length after CNN: 1080 -> ceil(1080/3)=360 -> ceil(360/2)=180 -> ceil(180/2)=90
CNN_KERNEL_SIZES = [15, 7, 3]
CNN_STRIDES      = [3, 2, 2]
CNN_CHANNELS     = [16, 24, 32]

# ── Transformer back-end ─────────────────────────────────────────────────────
NUM_ATTENTION_HEADS    = 2
TRANSFORMER_FF_DIM     = 64
NUM_TRANSFORMER_BLOCKS = 2


# ── 3.1 — 1D-CNN Feature Extractor ─────────────────────────────────────────────
def build_cnn_feature_extractor(input_tensor):
    """
    Stacks Conv1D layers to extract local ECG morphology features and
    progressively downsample the raw window into a short sequence.

    Args:
        input_tensor: Keras tensor of shape (batch, WINDOW_SIZE, 1)

    Returns:
        Keras tensor of shape (batch, reduced_seq_len, CNN_CHANNELS[-1])
    """
    x = input_tensor
    x = layers.Reshape((-1, 1, 1))(x)

    for kernel_size, stride, channels in zip(CNN_KERNEL_SIZES, CNN_STRIDES, CNN_CHANNELS):
        x = layers.Conv2D(
            filters=channels,
            kernel_size=(kernel_size, 1),
            strides=stride,
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Reshape((-1, CNN_CHANNELS[-1]))(x)

    return x


# ── 3.2 — Relative Positional Encoding ─────────────────────────────────────────
def relative_position_block(x, d_model, kernel_size=5, apply_qat=False):
    """A DepthwiseConv1D layer used to inject relative position information."""
    conv = layers.DepthwiseConv1D(kernel_size, padding="same", use_bias=True)
    if apply_qat:
        conv = tfmot.quantization.keras.quantize_annotate_layer(
            conv, quantize_config=DepthwiseConv1DQAT()
        )
    output = conv(x)
    return layers.Add()([x, output])


class DepthwiseConv1DQAT(tfmot.quantization.keras.QuantizeConfig):
    """QAT config for DepthwiseConv1D. per_axis must be False to avoid errors."""
    def get_weights_and_quantizers(self, layer):
        return [(layer.depthwise_kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.depthwise_kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
        return {}


# ── 3.3 — Transformer Encoder Block ────────────────────────────────────────────
class DotProductAttention(Layer):
    """Scaled Dot-Product Attention."""
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.softmax = layers.Activation('softmax')
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, key_dim=None, num_heads=None, seq_len=None):
        queries, keys, values = inputs
        # queries/keys/values: (batch, num_heads, seq_len, key_dim)
        scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, tf.float32))
        scores = tf.matmul(queries, keys, transpose_b=True)
        weights = tf.nn.softmax(scores * scale)
        weights = self.dropout(weights)
        return tf.matmul(weights, values)


class MultiHeadAttention(Layer):
    """Multi-Head Attention built from scratch so tfmot can quantize it."""
    def __init__(self, num_heads, key_dim, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = d_model
        self.attention = DotProductAttention()
        self.W_q = Dense(num_heads * key_dim)
        self.W_k = Dense(num_heads * key_dim)
        self.W_v = Dense(num_heads * key_dim)
        self.W_o = Dense(d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim, "d_model": self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        queries, keys, values = inputs
        batch = queries.shape[0]
        seq_len = queries.shape[1]
        reshape_batch = batch if batch is not None else -1

        q = self.W_q(queries)
        q = tf.reshape(q, (reshape_batch, seq_len, self.num_heads, self.key_dim))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = self.W_k(keys)
        k = tf.reshape(k, (reshape_batch, seq_len, self.num_heads, self.key_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.W_v(values)
        v = tf.reshape(v, (reshape_batch, seq_len, self.num_heads, self.key_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        o_reshaped = self.attention([q, k, v], key_dim=self.key_dim, num_heads=self.num_heads)

        o_transpose = tf.transpose(o_reshaped, perm=[0, 2, 1, 3])
        output = tf.reshape(o_transpose, (reshape_batch, seq_len, self.num_heads * self.key_dim))

        return self.W_o(output)


class MultiHeadAttentionQAT(tfmot.quantization.keras.QuantizeConfig):
    """Tells tfmot which parts of the custom attention head are quantizable."""
    def get_weights_and_quantizers(self, layer):
        return [(layer.W_q.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
                (layer.W_k.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
                (layer.W_v.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
                (layer.W_o.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.W_q.kernel = quantize_weights[0]
        layer.W_k.kernel = quantize_weights[1]
        layer.W_v.kernel = quantize_weights[2]
        layer.W_o.kernel = quantize_weights[3]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
        return {}


def transformer_encoder_block(x, d_model, num_heads, key_dim, ff_dim, apply_qat=False):
    """
    One Transformer encoder block: Multi-Head Self-Attention (+residual +LayerNorm)
    followed by a Feed-Forward sublayer (+residual +LayerNorm).
    """
    # --- Self-Attention sub-layer ---
    attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, d_model=d_model)

    if apply_qat:
        attn_output = tfmot.quantization.keras.quantize_annotate_layer(
            attn_layer, quantize_config=MultiHeadAttentionQAT()
        )([x, x, x])
    else:
        attn_output = attn_layer([x, x, x])

    attn_residual = layers.Add()([x, layers.Dropout(0.1)(attn_output)])

    # LayerNormalization isn't tfmot-compatible, so skip quant on this layer.
    norm1 = layers.LayerNormalization(epsilon=1e-6)
    if apply_qat:
        norm1 = tfmot.quantization.keras.quantize_annotate_layer(norm1, quantize_config=NoQuantizeConfig())
    x = norm1(attn_residual)

    # --- Feed-Forward sub-layer ---
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(d_model)(ff_output)   # project back to d_model

    ff_residual = layers.Add()([x, layers.Dropout(0.1)(ff_output)])

    norm2 = layers.LayerNormalization(epsilon=1e-6)
    if apply_qat:
        norm2 = tfmot.quantization.keras.quantize_annotate_layer(norm2, quantize_config=NoQuantizeConfig())
    x = norm2(ff_residual)
    return x


# ── 3.4 — Classification Head ──────────────────────────────────────────────────
def build_classification_head(x, num_classes=NUM_CLASSES):
    """
    Collapses the Transformer's (batch, seq_len, d_model) output into a
    single per-window vector, then projects it to class probabilities.
    """
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return x


# ── 4. Full Model Assembly ─────────────────────────────────────────────────────
class NoQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Tells tfmot to skip quantization for layers it can't handle automatically."""
    def get_weights_and_quantizers(self, layer): return []
    def get_activations_and_quantizers(self, layer): return []
    def set_quantize_weights(self, layer, quantize_weights): pass
    def set_quantize_activations(self, layer, quantize_activations): pass
    def get_output_quantizers(self, layer): return []
    def get_config(self): return {}


def build_model(window_size=WINDOW_SIZE, num_classes=NUM_CLASSES, apply_qat=False, batch_size=None):
    """
    Assembles the full CNN-Transformer model.

    Args:
        window_size: number of ECG samples per input window
        num_classes: number of output classes (2 for Normal vs. Abnormal)
        apply_qat:   if True, wrap custom layers so tfmot can apply
                     Quantization-Aware Training
        batch_size:  fixed batch size for the input layer (needed when
                     exporting a batch_size=1 model for on-device inference)
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(69)

    inputs = layers.Input(shape=(window_size, 1), batch_size=batch_size)

    # 1) CNN feature extraction + downsampling
    x = build_cnn_feature_extractor(inputs)
    d_model = CNN_CHANNELS[-1]      # Transformer's feature dim = CNN's final channel count

    # 2) Relative position information
    x = relative_position_block(x, d_model, kernel_size=5, apply_qat=apply_qat)

    # 3) Transformer encoder stack
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            x,
            d_model=d_model,
            num_heads=NUM_ATTENTION_HEADS,
            key_dim=d_model // NUM_ATTENTION_HEADS,
            ff_dim=TRANSFORMER_FF_DIM,
            apply_qat=apply_qat,
        )

    # 4) Classification head
    outputs = build_classification_head(x, num_classes)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_transformer_ecg_binary")
    return model
