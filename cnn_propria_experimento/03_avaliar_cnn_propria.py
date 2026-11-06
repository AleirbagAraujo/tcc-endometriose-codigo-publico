from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pipeline_utils import (
    CLASS_LABELS_PT,
    CLASS_NAMES,
    compute_binary_metrics,
    roc_curve_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a CNN propria no conjunto de teste."
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "dataset_cnn_propria",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "resultados_tcc"
        / "best_cnn_propria.keras",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "resultados_tcc",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Opcional. Por padrao, usa automaticamente o tamanho de entrada do modelo.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


def load_test_dataset(
    test_dir: Path,
    image_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, list[str]]:
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    if tuple(dataset.class_names) != CLASS_NAMES:
        raise RuntimeError(f"Ordem de classes inesperada: {dataset.class_names}")
    file_paths = list(dataset.file_paths)
    return dataset.prefetch(tf.data.AUTOTUNE), file_paths


def collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    batches = [labels.numpy().reshape(-1) for _, labels in dataset]
    return np.concatenate(batches).astype(int)


def resolve_image_size(model: tf.keras.Model, requested_size: int | None) -> int:
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) != 4:
        raise ValueError(f"Formato de entrada inesperado no modelo: {input_shape}")
    height, width = input_shape[1], input_shape[2]
    if height is None or width is None or height != width:
        raise ValueError(f"O modelo deve possuir entrada quadrada fixa: {input_shape}")
    model_size = int(height)
    if requested_size is not None and requested_size != model_size:
        raise ValueError(
            f"--img-size={requested_size} difere da entrada do modelo ({model_size})."
        )
    return model_size


def save_predictions(
    path: Path,
    file_paths: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "file",
                "true_label",
                "true_class",
                "prob_endometriosis",
                "predicted_label",
                "predicted_class",
            ],
        )
        writer.writeheader()
        for file_path, true_label, probability in zip(file_paths, y_true, y_prob):
            predicted_label = int(probability >= threshold)
            writer.writerow(
                {
                    "file": file_path,
                    "true_label": int(true_label),
                    "true_class": CLASS_NAMES[int(true_label)],
                    "prob_endometriosis": float(probability),
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES[predicted_label],
                }
            )


def save_confusion_matrix(path: Path, metrics: dict[str, float | int]) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    labels = [CLASS_LABELS_PT[name] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Oranges")
    fig.colorbar(image, ax=ax)
    ax.set_title("Matriz de confusao - CNN propria")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(path: Path, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, auc = roc_curve_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e17c05", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_title("Curva ROC - CNN propria")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    if not args.modelo.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {args.modelo}")

    class_names_path = args.modelo.parent / "class_names.json"
    if class_names_path.exists():
        saved_class_names = tuple(json.loads(class_names_path.read_text(encoding="utf-8")))
        if saved_class_names != CLASS_NAMES:
            raise RuntimeError(
                f"Classes do modelo {saved_class_names} diferem de {CLASS_NAMES}."
            )

    model = tf.keras.models.load_model(args.modelo)
    image_size = resolve_image_size(model, args.img_size)
    test_dataset, file_paths = load_test_dataset(
        args.dados / "test",
        image_size,
        args.batch_size,
    )
    y_true = collect_labels(test_dataset)
    y_prob = model.predict(test_dataset, verbose=1).reshape(-1)
    if not (len(file_paths) == len(y_true) == len(y_prob)):
        raise RuntimeError("Quantidade de arquivos, rotulos e predicoes nao coincide.")

    metrics = compute_binary_metrics(y_true, y_prob, args.threshold)
    predictions_path = args.saida / "predictions_test.csv"
    save_predictions(
        predictions_path,
        file_paths,
        y_true,
        y_prob,
        args.threshold,
    )
    with (args.saida / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    save_confusion_matrix(args.saida / "confusion_matrix.png", metrics)
    save_roc_curve(args.saida / "roc_curve.png", y_true, y_prob)

    print("\nAvaliacao finalizada.")
    print(f"Ordem das classes: {list(CLASS_NAMES)}")
    print("Classe positiva: endometriosis")
    print(f"Tamanho de entrada: {image_size} x {image_size}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predicoes: {predictions_path}")


if __name__ == "__main__":
    main()
