"""
FNN PRUNING – WEBSITE VERSION (NO FINE TUNING)

✔ Uses activation + gradient + variance
✔ Requires dataset (CSV)
✔ Protects final layer
✔ Produces STRUCTURALLY PRUNED MODEL
✔ Returns metrics for UI
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from sklearn.model_selection import train_test_split

# =========================================================
# 1) MODEL TYPE DETECTION
# =========================================================
def detect_model_type(model):
    out = model.layers[-1]
    units = getattr(out, "units", None)
    act = getattr(out, "activation", None)
    act_name = act.__name__ if act else None

    if units == 1 and act_name == "sigmoid":
        return "binary"
    if units == 1:
        return "regression"
    if units > 1:
        return "multiclass"
    return "unknown"


def get_loss(model_type):
    if model_type == "binary":
        return tf.keras.losses.BinaryCrossentropy()
    if model_type == "multiclass":
        return tf.keras.losses.SparseCategoricalCrossentropy()
    if model_type == "regression":
        return tf.keras.losses.MeanSquaredError()
    raise ValueError("Unsupported model type")

# =========================================================
# 2) CSV LOADER
# =========================================================
def load_csv(csv_path, model_type):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    if model_type == "binary":
        y = y.astype(np.float32).values.reshape(-1, 1)
    elif model_type == "multiclass":
        y = y.astype("category").cat.codes.values
    else:
        y = y.astype(np.float32).values.reshape(-1, 1)

    X = pd.get_dummies(X).astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 3) DENSE LAYERS
# =========================================================
def get_dense_layers(model):
    return [l for l in model.layers if isinstance(l, Dense)]

# =========================================================
# 4) IMPORTANCE SCORE (ACT + GRAD + VAR)
# =========================================================
def compute_importance(model, dense_layers, X, y, loss_fn, batches=20):
    stats = {l.name: [] for l in dense_layers}

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(64)

    for i, (xb, yb) in enumerate(dataset):
        if i >= batches:
            break

        x = tf.cast(xb, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            for layer in model.layers:
                x = layer(x)
                if layer in dense_layers:
                    tape.watch(x)
                    stats[layer.name].append(x)

            loss = tf.reduce_mean(loss_fn(yb, x))

        for layer in dense_layers:
            a = stats[layer.name][-1]     # activation
            g = tape.gradient(loss, a)    # gradient

            A = tf.reduce_mean(tf.abs(a), axis=0)
            V = tf.math.reduce_variance(a, axis=0)

            # ✅ FIX: ensure G is ALWAYS a tensor
            if g is None:
                G = tf.zeros_like(A)
            else:
                G = tf.reduce_mean(tf.abs(g), axis=0)

            stats[layer.name][-1] = (
                A.numpy(),
                G.numpy(),
                V.numpy()
            )

        del tape

    final = {}
    for name, values in stats.items():
        A, G, V = zip(*values)
        final[name] = (
            np.mean(A, axis=0),
            np.mean(G, axis=0),
            np.mean(V, axis=0),
        )

    return final

# =========================================================
# 5) MASK COMPUTATION
# =========================================================
def compute_masks(stats, keep_ratio, last_layer):
    masks = {}

    for name, (A, G, V) in stats.items():
        if name == last_layer:
            masks[name] = np.ones_like(A)
            continue

        score = 0.5*A + 0.3*G + 0.2*V
        k = max(1, int(len(score) * keep_ratio))
        idx = np.argsort(score)[-k:]

        mask = np.zeros_like(score)
        mask[idx] = 1
        masks[name] = mask

    return masks

# =========================================================
# 6) STRUCTURAL PRUNING
# =========================================================
def build_pruned_model(model, masks):
    inp = Input(shape=model.input_shape[1:])
    x = inp
    prev_idx = None

    for layer in model.layers:
        if isinstance(layer, Dense):
            W, b = layer.get_weights()

            if prev_idx is not None:
                W = W[prev_idx, :]

            if layer.name in masks:
                idx = np.where(masks[layer.name] == 1)[0]
            else:
                idx = np.arange(W.shape[1])

            W = W[:, idx]
            b = b[idx]

            new_layer = Dense(len(idx), activation=layer.activation, name=layer.name)
            x = new_layer(x)
            new_layer.set_weights([W, b])

            prev_idx = idx
        else:
            x = layer(x)

    return Model(inp, x)

# =========================================================
# 7) FLOPs (Dense only)
# =========================================================
def compute_dense_flops(model, masks=None):
    total = 0
    prev_keep = None

    for layer in model.layers:
        if isinstance(layer, Dense):
            W = layer.get_weights()[0]
            in_features = W.shape[0]
            out_features = W.shape[1]

            if prev_keep is not None:
                in_features = prev_keep

            if masks and layer.name in masks:
                out_features = int(np.sum(masks[layer.name]))

            total += in_features * out_features * 2

            if masks and layer.name in masks:
                prev_keep = int(np.sum(masks[layer.name]))
            else:
                prev_keep = out_features
        elif isinstance(layer, Flatten):
            prev_keep = None

    return total

# =========================================================
# 8) MAIN ENTRY FOR WEBSITE
# =========================================================
def prune_fnn_without_finetune(model_path, dataset_path, output_dir, keep_ratio=0.7):
    model = load_model(model_path)
    model_type = detect_model_type(model)
    loss_fn = get_loss(model_type)

    X_train, X_val, y_train, y_val = load_csv(dataset_path, model_type)

    dense_layers = get_dense_layers(model)
    last_dense = dense_layers[-1].name

    stats = compute_importance(model, dense_layers, X_train, y_train, loss_fn)
    masks = compute_masks(stats, keep_ratio, last_dense)

    pruned_model = build_pruned_model(model, masks)

    # ---------- METRICS ----------
    baseline_flops = compute_dense_flops(model)
    pruned_flops = compute_dense_flops(model, masks)

    # ---------- SAVE ----------
    base = os.path.splitext(os.path.basename(model_path))[0]
    pruned_path = os.path.join(output_dir, base + "_pruned.h5")
    pruned_model.save(pruned_path)

    report = {
        "baseline_flops": baseline_flops,
        "pruned_flops": pruned_flops,
        "reduction": (baseline_flops - pruned_flops) / baseline_flops
    }

    return pruned_path, report
