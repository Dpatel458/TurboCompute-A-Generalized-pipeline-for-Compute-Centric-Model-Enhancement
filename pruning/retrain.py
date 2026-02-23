import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class _CompatRandomFlip(tf.keras.layers.RandomFlip):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomRotation(tf.keras.layers.RandomRotation):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomZoom(tf.keras.layers.RandomZoom):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


_LOAD_CUSTOM_OBJECTS = {
    "RandomFlip": _CompatRandomFlip,
    "RandomRotation": _CompatRandomRotation,
    "RandomZoom": _CompatRandomZoom,
}

# =====================================================
# DATASET LOADERS
# =====================================================

def load_fnn_dataset(csv_path):
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path)

    # Features
    X = df.iloc[:, :-1].values.astype("float32")

    # Labels
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # One-hot if multiclass
    if len(np.unique(y)) > 2:
        y = tf.keras.utils.to_categorical(y)

    X = np.asarray(X, dtype=np.float32)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.asarray(y, dtype=np.float32)  # one-hot
    else:
        y = np.asarray(y, dtype=np.int32)    # sparse/binary class ids

    return X, y


def _infer_output_config(model):
    output_shape = model.output_shape
    if isinstance(output_shape, (list, tuple)) and output_shape and isinstance(output_shape[0], (list, tuple)):
        output_shape = output_shape[0]
    units = output_shape[-1]

    if units == 1:
        return {
            "loss": "binary_crossentropy",
            "label_mode": "binary",
            "multiclass": False,
        }

    return {
        "loss": "sparse_categorical_crossentropy",
        "label_mode": "int",
        "multiclass": True,
    }


def _safe_image_size(model, default_size=224):
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]
    h = input_shape[1] if len(input_shape) > 2 else None
    w = input_shape[2] if len(input_shape) > 2 else None

    if h is None or w is None:
        return (default_size, default_size)

    return (int(h), int(w))


def load_image_dataset(dataset_dir, model, batch_size=32, validation_split=0.2, seed=42):
    cfg = _infer_output_config(model)
    image_size = _safe_image_size(model)
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]
    channels = input_shape[-1] if len(input_shape) > 3 else 3
    color_mode = "grayscale" if channels == 1 else "rgb"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode=cfg["label_mode"],
        color_mode=color_mode,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode=cfg["label_mode"],
        color_mode=color_mode,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    scale = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (scale(x), y))
    val_ds = val_ds.map(lambda x, y: (scale(x), y))

    class_count = len(train_ds.class_names) if hasattr(train_ds, "class_names") else None

    return train_ds, val_ds, cfg, class_count


def load_text_dataset(dataset_path, model, batch_size=32):
    cfg = _infer_output_config(model)

    if dataset_path.lower().endswith(".npz"):
        data = np.load(dataset_path, allow_pickle=True)
        if "x" in data and "y" in data:
            X, y = data["x"], data["y"]
        else:
            keys = list(data.keys())
            if len(keys) < 2:
                raise ValueError("NPZ must contain at least two arrays for X and y")
            X, y = data[keys[0]], data[keys[1]]
    else:
        df = pd.read_csv(dataset_path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[-1]
        X = df[text_col].astype(str).values
        y = df[label_col].values

    class_count = None
    if isinstance(y, np.ndarray) and y.ndim > 1:
        class_count = y.shape[-1]

    if y.dtype.kind not in {"i", "u", "b"}:
        le = LabelEncoder()
        y = le.fit_transform(y)

    if not cfg["multiclass"]:
        y = y.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(min(len(X), 10000)).batch(batch_size)

    if class_count is None:
        class_count = int(len(np.unique(y)))

    return ds, cfg, class_count


def _validate_class_count(model, class_count):
    if class_count is None:
        return

    output_shape = model.output_shape
    if isinstance(output_shape, (list, tuple)) and output_shape and isinstance(output_shape[0], (list, tuple)):
        output_shape = output_shape[0]
    units = output_shape[-1]

    if units == 1:
        if class_count != 2:
            raise ValueError(
                f"Dataset has {class_count} classes but model output is binary (1 unit)."
            )
        return

    if units != class_count:
        raise ValueError(
            f"Dataset has {class_count} classes but model expects {units} classes."
        )


# =====================================================
# RETRAIN LOGIC
# =====================================================

def retrain_model(
    model_path,
    dataset_path,
    model_type,
    epochs,
    learning_rate,
    output_dir
):
    # ---------------- Load model ----------------
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=_LOAD_CUSTOM_OBJECTS
    )

    # ---------------- Load dataset ----------------
    if model_type == "fnn":
        X, y = load_fnn_dataset(dataset_path)
        output_shape = model.output_shape
        if isinstance(output_shape, (list, tuple)) and output_shape and isinstance(output_shape[0], (list, tuple)):
            output_shape = output_shape[0]
        units = output_shape[-1]

        if units == 1:
            loss = "binary_crossentropy"
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.reshape(-1)
            y = y.astype("float32")
        elif isinstance(y, np.ndarray) and y.ndim > 1:
            loss = "categorical_crossentropy"
            y = y.astype("float32")
        else:
            loss = "sparse_categorical_crossentropy"
            y = y.astype("int32")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )
        history = model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2
        )

    elif model_type in {"cnn", "resnet", "mobilenet", "efficientnet", "transfer"}:
        train_ds, val_ds, cfg, class_count = load_image_dataset(dataset_path, model)
        _validate_class_count(model, class_count)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=cfg["loss"],
            metrics=["accuracy"]
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    elif model_type == "transformer":
        ds, cfg, class_count = load_text_dataset(dataset_path, model)
        _validate_class_count(model, class_count)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=cfg["loss"],
            metrics=["accuracy"]
        )
        history = model.fit(ds, epochs=epochs)

    else:
        raise ValueError(f"Retraining not supported for {model_type} yet")

    # ---------------- Save retrained model ----------------
    retrained_name = (
        f"retrained_e{epochs}_lr{learning_rate}_"
        + os.path.basename(model_path)
    )
    retrained_path = os.path.join(output_dir, retrained_name)

    model.save(retrained_path)

    return retrained_path, history.history
