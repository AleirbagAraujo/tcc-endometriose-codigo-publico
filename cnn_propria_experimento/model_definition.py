from __future__ import annotations

import tensorflow as tf


def residual_block(
    inputs: tf.Tensor,
    filters: int,
    l2_strength: float,
    name: str,
) -> tf.Tensor:
    regularizer = tf.keras.regularizers.l2(l2_strength)
    shortcut = inputs

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{name}_conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"{name}_projection",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_projection_bn")(
            shortcut
        )

    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.Activation("relu", name=f"{name}_out")(x)


def multiscale_pooling(inputs: tf.Tensor) -> tf.Tensor:
    pooled = [inputs]
    for kernel_size in (5, 9, 13):
        pooled.append(
            tf.keras.layers.MaxPooling2D(
                pool_size=kernel_size,
                strides=1,
                padding="same",
                name=f"multiscale_pool_{kernel_size}",
            )(inputs)
        )
    return tf.keras.layers.Concatenate(name="multiscale_concatenate")(pooled)


def build_model(
    image_size: int = 224,
    dropout: float = 0.50,
    l2_strength: float = 1e-4,
    seed: int = 42,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(l2_strength)
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.10, seed=seed + 2),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed + 3),
            tf.keras.layers.RandomContrast(0.10, seed=seed + 4),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="stem_pool")(x)

    for index, filters in enumerate((64, 128, 256), start=1):
        x = residual_block(x, filters, l2_strength, name=f"residual_{index}")
        if filters != 256:
            x = tf.keras.layers.MaxPooling2D(2, name=f"residual_{index}_pool")(x)

    x = multiscale_pooling(x)
    x = tf.keras.layers.Conv2D(
        256,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="multiscale_projection",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="multiscale_projection_bn")(x)
    x = tf.keras.layers.Activation("relu", name="multiscale_projection_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, seed=seed + 5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="endometriosis_probability",
    )(x)
    return tf.keras.Model(inputs, outputs, name="endonet_cnn_propria")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
