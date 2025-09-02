import os
import socket

# Enable dynamic GPU memory allocation to avoid reserving all memory upfront.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf


def configure_environment():
    """Configure runtime settings to tolerate restricted environments."""
    if os.environ.get("RESTRICTED_ENV"):
        print("Restricted environment detected. Falling back to CPU and limiting GPU memory usage.")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            # Safe to ignore if GPUs are already hidden or unavailable
            pass

    try:
        socket.setdefaulttimeout(60)
        print("Network timeout set to 60 seconds.")
    except Exception as exc:
        print(f"Unable to adjust network settings: {exc}")


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(data, epochs: int = 10):
    configure_environment()
    model = build_model(data.shape[1:])
    model.fit(data, epochs=epochs)
    return model


if __name__ == "__main__":
    example_data = tf.random.normal([1000, 28, 28])
    train_model(example_data)
