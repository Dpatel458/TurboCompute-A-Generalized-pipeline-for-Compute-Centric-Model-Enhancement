"""
RESNET SOFT CHANNEL MASKING (NO DATASET / NO FINETUNE)

✔ Inserts ChannelMask after BN following Conv2D
✔ No training required
✔ Returns effective FLOPs report
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class _CompatRandomFlip(layers.RandomFlip):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomRotation(layers.RandomRotation):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomZoom(layers.RandomZoom):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


_LOAD_CUSTOM_OBJECTS = {
    "RandomFlip": _CompatRandomFlip,
    "RandomRotation": _CompatRandomRotation,
    "RandomZoom": _CompatRandomZoom,
}


# ---------------- FIND CONVS ----------------
def _find_convs(model):
    convs = []

    def walk(l):
        if isinstance(l, layers.Conv2D):
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
def _build_masked_resnet(base_model, masks, alpha):
    tensor_map = {}
    last_tensor = None

    for inp in base_model.inputs:
        x_in = inp
        try:
            if inp.shape[-1] == 1:
                x_in = layers.Lambda(
                    lambda t: tf.image.grayscale_to_rgb(t),
                    name="input_rgb"
                )(inp)
        except Exception:
            pass

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

        # Fix grayscale -> RGB mismatch for first conv if needed
        if isinstance(layer, layers.Conv2D):
            try:
                in_ch = layer.get_weights()[0].shape[2]
                mapped_ch = mapped.shape[-1]
                if mapped_ch == 1 and in_ch == 3:
                    mapped = layers.Lambda(
                        lambda t: tf.image.grayscale_to_rgb(t),
                        name=layer.name + "_rgb"
                    )(mapped)
            except Exception:
                pass

        x = layer(mapped)

        # Insert mask AFTER BatchNorm
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
    for l in _find_convs(model):
        if not l._inbound_nodes:
            continue
        _, h, w, cout = l.output.shape
        cin = l.input.shape[-1]
        k = l.kernel_size[0]
        total += h * w * cin * cout * k * k * 2
    return int(total)


def _effective_flops_from_masks(model, masks):
    total = 0
    for l in _find_convs(model):
        if l.name in masks and l._inbound_nodes:
            _, h, w, _ = l.output.shape
            cin = l.input.shape[-1]
            k = l.kernel_size[0]
            active = int(np.sum(masks[l.name]))
            total += h * w * cin * active * k * k * 2
    return int(total)

def format_flops(flops):
    if flops >= 1e9:
        return f"{flops / 1e9:.3f}"
    else:
        return f"{flops:.3f}"


# ---------------- MAIN ENTRY (NO DATASET) ----------------
def prune_resnet_mask(model_path, keep_ratio, output_dir, alpha=0.1):
    os.makedirs(output_dir, exist_ok=True)

    base = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=_LOAD_CUSTOM_OBJECTS
    )


    convs = _find_convs(base)
    masks = _make_masks(convs, keep_ratio)

    masked = _build_masked_resnet(base, masks, alpha=alpha)

    base_flops = _baseline_flops(base)
    eff_flops = _effective_flops_from_masks(base, masks)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    pruned_path = os.path.join(output_dir, base_name + "_masked.keras")
    masked.save(pruned_path)

    report = {
        "baseline_flops": format_flops(base_flops),
        "effective_flops": format_flops(eff_flops),
        "reduction": round((base_flops - eff_flops) / base_flops * 100, 2)
    }

    return pruned_path, report
