import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(
        self,
        max_len=None,
        vocab_size=None,
        embed_dim=None,
        maxlen=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_len is None and maxlen is not None:
            max_len = maxlen

        if max_len is None:
            raise ValueError("max_len or maxlen must be provided")

        self.max_len = int(max_len)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        self.token_emb = layers.Embedding(self.vocab_size, self.embed_dim)
        self.pos_emb = layers.Embedding(self.max_len, self.embed_dim)

    def call(self, x):
        pos = tf.range(start=0, limit=tf.shape(x)[-1])
        return self.token_emb(x) + self.pos_emb(pos)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "maxlen": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return cfg


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

        self.last_attn = None

    def call(self, x, training=None):
        attn = self.att(x, x, training=training)
        self.last_attn = attn
        x = self.ln1(x + self.drop1(attn, training=training))
        ffn = self.ffn(x, training=training)
        return self.ln2(x + self.drop2(ffn, training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return cfg


TransformerBlock = TransformerEncoder


@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskedTransformer(tf.keras.Model):
    def __init__(self, base_model, masks):
        super().__init__()
        self.base = base_model
        self.mask_values = {
            layer.name: np.asarray(mask, dtype="float32")
            for layer, mask in masks.items()
        }

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.base.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            if isinstance(layer, TransformerEncoder):
                attn = layer.att(x, x, training=training)

                if layer.name in self.mask_values:
                    heads = layer.att.num_heads
                    width = attn.shape[-1]
                    head_dim = width // heads
                    mask = self.mask_values[layer.name]

                    attn = tf.reshape(attn, (-1, tf.shape(attn)[1], heads, head_dim))
                    attn = attn * mask[None, None, :, None]
                    attn = tf.reshape(attn, (-1, tf.shape(attn)[1], width))

                x = layer.ln1(x + layer.drop1(attn, training=training))
                x = layer.ln2(x + layer.drop2(layer.ffn(x), training=training))
            else:
                x = layer(x, training=training)

        return x

    def get_config(self):
        return {
            "base_model": tf.keras.utils.serialize_keras_object(self.base),
            "masks": {name: mask.tolist() for name, mask in self.mask_values.items()},
        }

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.utils.deserialize_keras_object(config["base_model"])
        masks = {
            layer: np.asarray(config["masks"][layer.name], dtype="float32")
            for layer in base_model.layers
            if layer.name in config["masks"]
        }
        return cls(base_model=base_model, masks=masks)


def _extract_dtype(tensor_or_tensors):
    if isinstance(tensor_or_tensors, (list, tuple)):
        for item in tensor_or_tensors:
            dtype = _extract_dtype(item)
            if dtype is not None:
                return dtype
        return None
    return getattr(tensor_or_tensors, "dtype", None)


def _strip_transformer(model):
    model_input_dtype = _extract_dtype(model.input)
    if model_input_dtype in (tf.int32, tf.int64):
        return model

    for layer in model.layers:
        try:
            layer_input_dtype = _extract_dtype(layer.input)
            if layer_input_dtype in (tf.int32, tf.int64):
                return tf.keras.Model(layer.input, model.output)
        except Exception:
            continue

    seq_len = _infer_seq_len(model)
    token_input = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
    return tf.keras.Model(token_input, model(token_input))


def _get_all_layers(model):
    out = []
    for layer in model.layers:
        out.append(layer)
        if isinstance(layer, tf.keras.Model):
            out.extend(_get_all_layers(layer))
    return out


def _compute_attention_head_stats(model):
    stats = {}
    for block in _get_all_layers(model):
        if not isinstance(block, TransformerEncoder):
            continue

        num_heads = block.att.num_heads
        head_scores = np.zeros(num_heads, dtype="float32")

        for weight in block.att.weights:
            arr = weight.numpy()

            if arr.ndim >= 2 and arr.shape[-2] == num_heads:
                reduce_axes = tuple(i for i in range(arr.ndim) if i != arr.ndim - 2)
                head_scores += np.mean(np.abs(arr), axis=reduce_axes)
            elif arr.ndim >= 1 and arr.shape[0] == num_heads:
                reduce_axes = tuple(range(1, arr.ndim))
                head_scores += np.mean(np.abs(arr), axis=reduce_axes)

        if np.allclose(head_scores, 0):
            head_scores = np.ones(num_heads, dtype="float32")

        stats[block] = head_scores

    return stats


def _compute_importance_mask(stats, keep_ratio):
    masks = {}
    for block, score in stats.items():
        keep = max(1, int(len(score) * keep_ratio))
        threshold = np.partition(score, -keep)[-keep]
        masks[block] = (score >= threshold).astype(np.float32)
    return masks


def _attention_flops(seq_len, embed_dim):
    return 4 * seq_len * embed_dim * embed_dim + 2 * seq_len * seq_len * embed_dim


def _infer_seq_len(model, default_len=200):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) > 1 and input_shape[1] is not None:
        return int(input_shape[1])
    return default_len


def _transformer_model_flops(model):
    flops = 0
    seq_len = _infer_seq_len(model)
    for layer in _get_all_layers(model):
        if isinstance(layer, TransformerEncoder):
            embed_dim = layer.att.key_dim * layer.att.num_heads
            flops += _attention_flops(seq_len, embed_dim)
    return flops


def _effective_transformer_flops(model, masks):
    flops = 0
    seq_len = _infer_seq_len(model)
    for layer in _get_all_layers(model):
        if isinstance(layer, TransformerEncoder):
            embed_dim = layer.att.key_dim * layer.att.num_heads
            heads = layer.att.num_heads
            kept = int(np.sum(masks.get(layer, np.ones(heads))))
            flops += _attention_flops(seq_len, embed_dim) * (kept / heads)
    return flops


def prune_transformer_heads(model_path, keep_ratio, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerBlock": TransformerEncoder,
        },
        compile=False,
    )

    transformer = _strip_transformer(model)
    stats = _compute_attention_head_stats(transformer)
    if not stats:
        raise ValueError("No TransformerEncoder blocks were found in the model")

    masks = _compute_importance_mask(stats, keep_ratio)

    masked_model = MaskedTransformer(transformer, masks)

    baseline_flops = _transformer_model_flops(transformer)
    effective_flops = _effective_transformer_flops(transformer, masks)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    pruned_path = os.path.join(output_dir, base_name + "_transformer_pruned.keras")
    masked_model.save(pruned_path)

    report = {
        "baseline_flops": int(baseline_flops),
        "effective_flops": int(effective_flops),
        "reduction": round((1 - (effective_flops / baseline_flops)) * 100, 2),
        "heads_pruned": int(
            sum(
                layer.att.num_heads - int(np.sum(mask))
                for layer, mask in masks.items()
            )
        ),
        "heads_kept": int(sum(int(np.sum(mask)) for mask in masks.values())),
    }

    return pruned_path, report
