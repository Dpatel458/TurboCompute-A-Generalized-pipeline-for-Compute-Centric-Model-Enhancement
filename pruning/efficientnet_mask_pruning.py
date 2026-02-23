"""
EFFICIENTNET SOFT CHANNEL MASKING (NO DATASET / NO FINETUNE)

✔ Masks pointwise Conv2D (1x1) only
✔ Skips DepthwiseConv2D
✔ Uses 3-channel dummy input to build graph safely
✔ Returns effective FLOPs report
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ---------------- FIND PRUNABLE CONVS ----------------
def _find_prunable_convs(model):
    convs = []

    def walk(l):
        if isinstance(l, layers.Conv2D):
            if l.kernel_size == (1, 1):
                convs.append(l)
        if isinstance(l, tf.keras.Model):
            for x in l.layers:
                walk(x)

    walk(model)
    return convs


# ---------------- SOFT CHANNEL MASK ----------------
@tf.keras.utils.register_keras_serializable()
class ChannelMask(layers.Layer):
    def __init__(self, mask, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        mask = np.asarray(mask, dtype="float32")
        mask = np.where(mask > 0, 1.0, alpha)
        mask = mask / np.mean(mask)
        self.mask_init = mask
        self.alpha = alpha

    def build(self, input_shape):
        self.mask = self.add_weight(
            name="mask",
            shape=(input_shape[-1],),
            initializer=tf.constant_initializer(self.mask_init),
            trainable=False
        )

    def call(self, x):
        return x * self.mask[None, None, None, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "mask": self.mask_init.tolist(),
            "alpha": self.alpha
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        mask = config.pop("mask")
        return cls(mask=mask, **config)


# ---------------- MASK CREATION ----------------
def _make_masks(convs, keep_ratio):
    masks = {}
    for l in convs:
        W = l.kernel.numpy()
        scores = np.mean(np.abs(W), axis=(0, 1, 2))
        k = max(1, int(len(scores) * keep_ratio))
        thr = np.partition(scores, -k)[-k]
        mask = (scores >= thr).astype("float32")
        masks[l.name] = mask
    return masks


# ---------------- GRAPH-SAFE MASK INSERT ----------------
def _build_masked_efficientnet(base_model, masks, alpha):
    tensor_map = {}
    last_tensor = None

    def _to_rgb(t):
        if t.shape[-1] == 1:
            t = tf.image.grayscale_to_rgb(t)
            t.set_shape(t.shape[:-1].concatenate(3))
        return t

    for inp in base_model.inputs:
        x_in = inp
        if inp.shape[-1] == 1:
            x_in = layers.Lambda(
                _to_rgb,
                name="input_rgb",
                output_shape=lambda s: s[:-1] + (3,)
            )(inp)

        tensor_map[inp] = x_in
        last_tensor = x_in

    for layer in base_model.layers:
        if isinstance(layer, layers.InputLayer):
            continue

        inbound = layer.input
        if isinstance(inbound, list):
            mapped = [tensor_map.get(x, x) for x in inbound]
        else:
            mapped = tensor_map.get(inbound, inbound)

        x = layer(mapped)

        # Insert mask AFTER BatchNorm if previous layer was Conv2D(1x1)
        if isinstance(layer, layers.BatchNormalization):
            prev = layer.input._keras_history[0]
            if isinstance(prev, layers.Conv2D) and prev.name in masks:
                x = ChannelMask(
                    masks[prev.name],
                    alpha=alpha,
                    name=prev.name + "_mask"
                )(x)

        # Handle layers with multiple outputs
        if isinstance(layer.output, (list, tuple)):
            for out in layer.output:
                tensor_map[out] = x
        else:
            tensor_map[layer.output] = x
        last_tensor = x

    return tf.keras.Model(
        inputs=base_model.inputs,
        outputs=last_tensor,
        name=base_model.name + "_masked"
    )


# ---------------- FLOPs ----------------
def _baseline_flops(model):
    total = 0
    for l in _find_prunable_convs(model):
        if not l._inbound_nodes:
            continue
        _, h, w, cout = l.output.shape
        cin = l.input.shape[-1]
        total += h * w * cin * cout * 2
    return int(total)


def _effective_flops_from_masks(model, masks):
    total = 0
    for l in _find_prunable_convs(model):
        if l.name in masks and l._inbound_nodes:
            _, h, w, _ = l.output.shape
            cin = l.input.shape[-1]
            active = int(np.sum(masks[l.name]))
            total += h * w * cin * active * 2
    return int(total)


# ---------------- MAIN ENTRY (NO DATASET) ----------------
def _warmup_with_dataset(model, dataset_dir):
    if not dataset_dir:
        return

    h, w = model.input_shape[1], model.input_shape[2]
    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=(h, w),
        batch_size=1,
        shuffle=False
    )
    scale = layers.Rescaling(1.0 / 255.0)
    ds = ds.map(lambda x, y: (scale(x), y))

    for x, _ in ds.take(1):
        _ = model(x, training=False)


def prune_efficientnet_mask(model_path, keep_ratio, output_dir, alpha=0.1, dataset_dir=None):
    os.makedirs(output_dir, exist_ok=True)

    base = tf.keras.models.load_model(model_path, compile=False)

    # Warm up with real RGB data if provided (builds graph safely)
    _warmup_with_dataset(base, dataset_dir)

    convs = _find_prunable_convs(base)
    masks = _make_masks(convs, keep_ratio)

    masked = _build_masked_efficientnet(base, masks, alpha=alpha)

    base_flops = _baseline_flops(base)
    eff_flops = _effective_flops_from_masks(base, masks)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    pruned_path = os.path.join(output_dir, base_name + "_masked.keras")
    masked.save(pruned_path)

    report = {
        "baseline_flops": base_flops,
        "effective_flops": eff_flops,
        "reduction": round((base_flops - eff_flops) / base_flops * 100, 2)
    }

    return pruned_path, report
