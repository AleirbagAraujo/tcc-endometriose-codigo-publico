from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_definition import build_model, compile_model
from pipeline_utils import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a CNN propria para classificacao binaria de endometriose."
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "dataset_cnn_propria",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "resultados_tcc",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.50)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_dataset(
    folder: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if tuple(dataset.class_names) != CLASS_NAMES:
        raise RuntimeError(
            f"Ordem de classes inesperada: {dataset.class_names}; esperado: {CLASS_NAMES}"
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def save_model_summary(model: tf.keras.Model, path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.saida.mkdir(parents=True, exist_ok=True)

    train_dir = args.dados / "train"
    val_dir = args.dados / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset nao preparado. Execute primeiro 01_preparar_dataset.py."
        )

    train_dataset = load_dataset(
        train_dir,
        args.img_size,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = load_dataset(
        val_dir,
        args.img_size,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = build_model(
        image_size=args.img_size,
        dropout=args.dropout,
        l2_strength=args.l2,
        seed=args.seed,
    )
    compile_model(model, learning_rate=args.learning_rate)

    best_model_path = args.saida / "best_cnn_propria.keras"
    last_model_path = args.saida / "last_cnn_propria.keras"
    history_path = args.saida / "history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            last_model_path,
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    config = {
        "architecture": "CNN propria com blocos residuais e pooling multiescala",
        "dataset": str(args.dados),
        "class_names": list(CLASS_NAMES),
        "positive_class": "endometriosis",
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "l2": args.l2,
        "early_stopping_monitor": "val_auc",
        "early_stopping_patience": args.patience,
        "seed": args.seed,
        "deterministic_operations": True,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_factor": 0.08,
            "zoom_factor": 0.10,
            "translation_factor": 0.05,
            "contrast_factor": 0.10,
        },
    }
    with (args.saida / "config_treinamento.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    with (args.saida / "class_names.json").open("w", encoding="utf-8") as stream:
        json.dump(list(CLASS_NAMES), stream, ensure_ascii=False, indent=2)
    save_model_summary(model, args.saida / "model_summary.txt")

    print(f"Ordem das classes: {list(CLASS_NAMES)}")
    print("Classe positiva: endometriosis")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    completed_epochs = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_auc"])) + 1
    best_val_auc = float(np.max(history.history["val_auc"]))
    config["completed_epochs"] = completed_epochs
    config["best_epoch_by_val_auc"] = best_epoch
    config["best_val_auc"] = best_val_auc
    with (args.saida / "config_treinamento.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    print("\nTreinamento finalizado.")
    print(f"Epocas executadas: {completed_epochs}")
    print(f"Melhor epoca por val_auc: {best_epoch}")
    print(f"Melhor val_auc: {best_val_auc:.4f}")
    print(f"Melhor modelo: {best_model_path}")
    print(f"Modelo final: {last_model_path}")
    print(f"Historico: {history_path}")


if __name__ == "__main__":
    main()
