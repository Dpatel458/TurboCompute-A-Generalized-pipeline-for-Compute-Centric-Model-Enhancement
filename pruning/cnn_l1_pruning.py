import os
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ======================================================
# SAFE MODEL LOADING
# ======================================================
def safe_load_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        with h5py.File(path, "r") as f:
            raw = f["model_config"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            cfg = json.loads(raw)
            model = tf.keras.models.model_from_json(json.dumps(cfg))
            model.load_weights(path)
            return model

# ======================================================
# L1 IMPORTANCE (Conv filters only)
# ======================================================
def compute_l1_importance(model):
    scores = {}
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            W = layer.get_weights()[0]
            s = np.sum(np.abs(W), axis=(0, 1, 2))
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            scores[layer.name] = s
    return scores

# ======================================================
# MASKS
# ======================================================
def make_masks(scores, keep_ratio):
    masks = {}
    for name, s in scores.items():
        k = max(1, int(len(s) * keep_ratio))
        thresh = np.partition(s, -k)[-k]
        masks[name] = (s >= thresh).astype(np.float32)
    return masks

# ======================================================
# STRUCTURAL PRUNING
# ======================================================
def structural_prune(model, masks, input_shape):
    seq = models.Sequential(name="pruned_model")
    seq.add(layers.InputLayer(shape=input_shape))

    prev_keep = None
    after_flatten = False

    for layer in model.layers:

        # -------- Conv2D (pruned) --------
        if isinstance(layer, layers.Conv2D):
            W, b = layer.get_weights()

            if prev_keep is not None:
                W = W[:, :, prev_keep, :]

            mask = masks[layer.name]
            keep = np.where(mask == 1)[0]
            if keep.size == 0:
                keep = np.array([np.argmax(mask)])

            Wn = W[:, :, :, keep]
            bn = b[keep]

            new_conv = layers.Conv2D(
                filters=len(keep),
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                use_bias=True,
            )

            seq.add(new_conv)
            new_conv.set_weights([Wn, bn])
            prev_keep = keep
            continue

        # -------- Flatten / GAP --------
        if isinstance(layer, (layers.Flatten, layers.GlobalAveragePooling2D)):
            seq.add(layer.__class__.from_config(layer.get_config()))
            prev_keep = None
            after_flatten = True
            continue

        # -------- Dense --------
        if isinstance(layer, layers.Dense):
            new_dense = layers.Dense.from_config(layer.get_config())
            seq.add(new_dense)

            # Do NOT copy weights for first dense after flatten
            if not after_flatten:
                try:
                    new_dense.set_weights(layer.get_weights())
                except ValueError:
                    pass

            after_flatten = False
            prev_keep = None
            continue

        # -------- Other layers --------
        cloned = layer.__class__.from_config(layer.get_config())
        seq.add(cloned)
        if layer.get_weights():
            cloned.set_weights(layer.get_weights())

    return seq

# ======================================================
# TRUE FORWARD-PASS GFLOPs (SINGLE INFERENCE)
# ======================================================
def _conv_out(size, k, s, padding):
    if padding == "same":
        return int(np.ceil(size / s))
    return int(np.floor((size - k + 1) / s))

def model_gflops(model, input_shape):
    total_flops = 0
    shape = list(input_shape)

    for layer in model.layers:

        # -------- Conv2D --------
        if isinstance(layer, layers.Conv2D):
            kh, kw, cin, cout = layer.get_weights()[0].shape
            sh, sw = layer.strides
            h, w, _ = shape

            hout = _conv_out(h, kh, sh, layer.padding)
            wout = _conv_out(w, kw, sw, layer.padding)

            total_flops += hout * wout * cin * cout * kh * kw * 2
            shape = [hout, wout, cout]

        # -------- DepthwiseConv --------
        elif isinstance(layer, layers.DepthwiseConv2D):
            kh, kw, cin, _ = layer.get_weights()[0].shape
            sh, sw = layer.strides
            h, w, c = shape

            hout = _conv_out(h, kh, sh, layer.padding)
            wout = _conv_out(w, kw, sw, layer.padding)

            total_flops += hout * wout * cin * kh * kw * 2
            shape = [hout, wout, cin]

        # -------- Pooling --------
        elif isinstance(layer, (layers.MaxPooling2D, layers.AveragePooling2D)):
            sh, sw = layer.strides
            h, w, c = shape
            shape = [int(np.ceil(h / sh)), int(np.ceil(w / sw)), c]

        # -------- Global Pool --------
        elif isinstance(layer, layers.GlobalAveragePooling2D):
            shape = [shape[2]]

        # -------- Flatten --------
        elif isinstance(layer, layers.Flatten):
            shape = [np.prod(shape)]

        # -------- Dense --------
        elif isinstance(layer, layers.Dense):
            total_flops += shape[0] * layer.units * 2
            shape = [layer.units]

    return total_flops / 1e9  # GFLOPs

# ======================================================
# MAIN ENTRY (USED BY WEBSITE)
# ======================================================
def prune_cnn(model_path, keep_ratio, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = safe_load_model(model_path)
    input_shape = model.input_shape[1:]

    scores = compute_l1_importance(model)
    masks = make_masks(scores, keep_ratio)

    pruned = structural_prune(model, masks, input_shape)

    base_gflops = model_gflops(model, input_shape)
    pruned_gflops = model_gflops(pruned, input_shape)

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    base_path = os.path.join(output_dir, base_name + "_baseline.keras")
    pruned_path = os.path.join(output_dir, base_name + "_pruned.keras")

    model.save(base_path)
    pruned.save(pruned_path)

    return {
        "base_gflops": round(base_gflops, 4),
        "pruned_gflops": round(pruned_gflops, 4),
        "reduction": round((1 - pruned_gflops / base_gflops) * 100, 2),
        "pruned_model_path": pruned_path
    }
