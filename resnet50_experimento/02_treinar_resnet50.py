from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = ["no_endometriosis", "endometriosis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 model for binary endometriosis classification."
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("resnet50_experimento") / "resultados" / "dataset_resnet50",
        help="Folder containing train, val, and test splits.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resnet50_experimento") / "resultados" / "resultados_tcc",
        help="Output folder for model and logs.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs-head", type=int, default=12, help="Head training epochs.")
    parser.add_argument("--epochs-fine", type=int, default=12, help="Fine-tuning epochs.")
    parser.add_argument("--fine-tune-layers", type=int, default=30, help="Train the last N ResNet layers.")
    parser.add_argument("--lr-head", type=float, default=1e-4, help="Learning rate for head training.")
    parser.add_argument("--lr-fine", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--dropout", type=float, default=0.30, help="Dropout before final layer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(
    folder: Path,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def configure_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    return dataset.prefetch(autotune)


def build_model(
    img_size: int,
    dropout: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="endometriosis_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_endometriosis")
    return model, base_model


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


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    dataframe = pd.DataFrame(history.history)
    dataframe.to_csv(path, index=False)


def unfreeze_for_fine_tuning(base_model: tf.keras.Model, fine_tune_layers: int) -> None:
    base_model.trainable = True

    if fine_tune_layers <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return

    freeze_until = max(0, len(base_model.layers) - fine_tune_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Batch normalization statistics are fragile during fine-tuning with small datasets.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.saida.mkdir(parents=True, exist_ok=True)

    image_size = (args.img_size, args.img_size)
    train_ds = load_dataset(
        args.dados / "train",
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = load_dataset(
        args.dados / "val",
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    train_ds = configure_dataset(train_ds)
    val_ds = configure_dataset(val_ds)

    model, base_model = build_model(img_size=args.img_size, dropout=args.dropout)

    with (args.saida / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, ensure_ascii=False, indent=2)

    checkpoint_path = args.saida / "best_resnet50.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("Stage 1: training only the new classification head.")
    compile_model(model, args.lr_head)
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )
    save_history(history_head, args.saida / "history_head.csv")

    print("Stage 2: fine-tuning the last ResNet50 layers.")
    unfreeze_for_fine_tuning(base_model, args.fine_tune_layers)
    compile_model(model, args.lr_fine)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_fine,
        callbacks=callbacks,
    )
    save_history(history_fine, args.saida / "history_fine.csv")

    model.save(args.saida / "last_resnet50.keras")
    print()
    print("Training finished.")
    print(f"Best model: {checkpoint_path}")
    print(f"Last model: {args.saida / 'last_resnet50.keras'}")


if __name__ == "__main__":
    main()
