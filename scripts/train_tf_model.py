#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ember / Neural — TensorFlow Training Utility

Features
- Safe environment config (GPU growth, restricted env behavior, socket timeout)
- Deterministic / reproducible runs (seeded)
- Optional mixed precision (auto if GPU supports float16/bfloat16)
- MNIST dataset loader with offline fallback to synthetic data
- tf.data input pipeline with caching, shuffling, prefetching
- Simple MLP or small ConvNet (select via --model {mlp,conv})
- Cosine learning-rate schedule + EarlyStopping + ModelCheckpoint
- TensorBoard logging
- CLI flags for epochs, batch size, learning rate, output dir, etc.
- Exports SavedModel and a JSON summary of metrics

Usage
-----
python scripts/train_tf_model.py --model conv --epochs 5 --batch 128 --outdir runs/tf
python scripts/train_tf_model.py --no-download  # force synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Tuple, Optional

# ---- Environment knobs (set before TF import) -------------------------------
# Avoids TF grabbing all GPU memory up front.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# Quieter logs unless debugging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Environment & determinism
# -----------------------------------------------------------------------------
def configure_environment(restricted_env: bool = False, timeout_s: int = 60, seed: Optional[int] = 42) -> None:
    """Configure runtime settings to tolerate restricted environments."""
    if restricted_env or os.environ.get("RESTRICTED_ENV"):
        print("[env] Restricted environment detected: preferring CPU, limiting GPU usage.")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    try:
        socket.setdefaulttimeout(timeout_s)
        print(f"[env] Network timeout set to {timeout_s} seconds.")
    except Exception as exc:
        print(f"[env] Unable to adjust network settings: {exc}")

    # Deterministic
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
            print("[env] TensorFlow deterministic ops enabled.")
        except Exception:
            print("[env] Deterministic ops not available on this TF build.")

    # (Optional) enable mixed precision if GPU supports it
    try:
        if tf.config.list_physical_devices("GPU"):
            # prefer bfloat16 if available (Ampere+ or TPUs), else float16
            policy = tf.keras.mixed_precision.Policy("mixed_bfloat16" if tf.config.experimental.get_device_details(
                tf.config.list_physical_devices("GPU")[0]).get("compute_capability", (0, 0))[0] >= 8 else "mixed_float16"
            )
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[env] Mixed precision policy set: {policy.name}")
    except Exception as e:
        print(f"[env] Mixed precision not enabled: {e}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def _make_synthetic(num_examples: int = 60_000, image_shape=(28, 28), num_classes: int = 10):
    x = np.random.rand(num_examples, *image_shape).astype("float32")
    y = np.random.randint(0, num_classes, size=(num_examples,), dtype="int32")
    return (x, y)


def load_data(no_download: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST from Keras datasets. If download/cache is unavailable or --no-download,
    falls back to synthetic data so training still works offline.
    """
    if no_download:
        print("[data] Using synthetic data (--no-download).")
        train = _make_synthetic(60_000)
        test = _make_synthetic(10_000)
        return train, test

    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("[data] Loaded MNIST from keras.datasets (using local cache or download).")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"[data] MNIST load failed ({e}). Falling back to synthetic data.")
        train = _make_synthetic(60_000)
        test = _make_synthetic(10_000)
        return train, test


def make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    cache: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create tf.data pipelines with sensible defaults."""
    AUTOTUNE = tf.data.AUTOTUNE

    def _prep(x, y):
        # Normalize to [0,1] and add channel dim
        x = tf.cast(x, tf.float32) / 255.0 if tf.reduce_max(x) > 1.0 else tf.cast(x, tf.float32)
        if tf.rank(x) == 3:  # [B,H,W] -> [B,H,W,1]
            x = tf.expand_dims(x, -1)
        return x, tf.cast(y, tf.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(_prep, num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(_prep, num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = (
        train_ds.shuffle(min(len(x_train), 10_000))
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.batch(max(32, batch_size), drop_remainder=False).prefetch(AUTOTUNE)

    return train_ds, val_ds


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def build_mlp(input_shape=(28, 28, 1), num_classes: int = 10, width: int = 128, depth: int = 2) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for _ in range(depth):
        x = tf.keras.layers.Dense(width, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # ensure float32 output
    return tf.keras.Model(inputs, outputs, name="mlp")


def build_conv(input_shape=(28, 28, 1), num_classes: int = 10, width: int = 32) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(width * 2, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="convnet")


def compile_model(model: tf.keras.Model, lr: float) -> None:
    # Cosine decay schedule with a small floor (10% of base lr)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=200,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.10,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------
def train(
    model_name: str = "mlp",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    outdir: str = "runs/tf",
    no_download: bool = False,
    restricted_env: bool = False,
    seed: Optional[int] = 42,
) -> dict:
    configure_environment(restricted_env=restricted_env, seed=seed)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tb_dir = out / f"tb_{int(time.time())}"
    ckpt_path = out / "best.keras"

    # Data
    (x_train, y_train), (x_test, y_test) = load_data(no_download=no_download)

    # Split off validation from train (e.g., 55k/5k)
    val_size = min(5_000, len(x_train) // 10)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    train_ds, val_ds = make_datasets(x_train, y_train, x_val, y_val, batch_size=batch_size, cache=True)

    # Model
    if model_name.lower() == "conv":
        model = build_conv()
    else:
        model = build_mlp()
    compile_model(model, lr=lr)
    model.summary()

    # Callbacks
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), profile_batch=0),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=cbs,
    )

    # Evaluate on test
    # Build test dataset quickly
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(
        lambda x, y: (tf.cast(tf.expand_dims(x, -1), tf.float32) / 255.0, tf.cast(y, tf.int32))
    ).batch(max(32, batch_size)).prefetch(tf.data.AUTOTUNE)

    test_metrics = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names  # ["loss", "accuracy"]
    test_report = dict(zip([f"test_{m}" for m in metric_names], [float(m) for m in test_metrics]))

    # Export SavedModel
    export_dir = out / "saved_model"
    tf.saved_model.save(model, str(export_dir))
    print(f"[save] SavedModel -> {export_dir}")
    if ckpt_path.exists():
        print(f"[save] Best Keras model -> {ckpt_path}")

    # Persist metrics
    payload = {
        "config": {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "outdir": str(out.resolve()),
            "no_download": no_download,
            "restricted_env": restricted_env,
            "seed": seed,
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "test": test_report,
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] Metrics -> {out/'metrics.json'}")

    return payload


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small TF model (MNIST or synthetic fallback).")
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "conv"], help="Model type")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    p.add_argument("--outdir", type=str, default="runs/tf", help="Output directory for logs/models")
    p.add_argument("--no-download", action="store_true", help="Use synthetic data instead of downloading MNIST")
    p.add_argument("--restricted-env", action="store_true", help="Behave as if in a restricted environment")
    p.add_argument("--seed", type=int, default=42, help="Random seed (set to -1 for non-deterministic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = None if (args.seed is not None and int(args.seed) < 0) else int(args.seed)
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        outdir=args.outdir,
        no_download=args.no_download,
        restricted_env=args.restricted_env,
        seed=seed,
    )


if __name__ == "__main__":
    main()