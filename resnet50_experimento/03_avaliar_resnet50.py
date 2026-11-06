from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


CLASS_NAMES = ["no_endometriosis", "endometriosis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ResNet50 model on the test split."
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("resnet50_experimento") / "resultados" / "dataset_resnet50",
        help="Folder containing train, val, and test splits.",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path ("resnet50_experimento") / "resultados" / "resultados_tcc" / "best_resnet50.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resnet50_experimento") / "resultados" / "resultados_tcc",
        help="Output folder for evaluation files.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold.")
    return parser.parse_args()


def load_test_dataset(
    test_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> tuple[tf.data.Dataset, list[str]]:
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    file_paths = list(getattr(dataset, "file_paths", []))
    return dataset.prefetch(tf.data.AUTOTUNE), file_paths


def collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[np.ndarray] = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy().reshape(-1))
    return np.concatenate(labels).astype(int)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    scores = np.concatenate([positives, negatives])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)

    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    positive_ranks = ranks[:n_pos]
    auc = (positive_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = safe_divide(tp + tn, len(y_true))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    auc = binary_auc(y_true, y_prob)

    return {
        "threshold": threshold,
        "total": int(len(y_true)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "auc": auc,
    }


def save_predictions(
    path: Path,
    file_paths: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
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
        for file_path, true_label, prob in zip(file_paths, y_true, y_prob):
            predicted_label = int(prob >= threshold)
            writer.writerow(
                {
                    "file": file_path,
                    "true_label": int(true_label),
                    "true_class": CLASS_NAMES[int(true_label)],
                    "prob_endometriosis": float(prob),
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

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    test_ds, file_paths = load_test_dataset(
        args.dados / "test",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    model = tf.keras.models.load_model(args.modelo)

    y_true = collect_labels(test_ds)
    y_prob = model.predict(test_ds).reshape(-1)
    metrics = compute_metrics(y_true, y_prob, args.threshold)

    save_predictions(
        args.saida / "predictions_test.csv",
        file_paths=file_paths,
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
    )

    with (args.saida / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_confusion_matrix(args.saida / "confusion_matrix.png", metrics)

    print("Evaluation finished.")
    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {args.saida / 'predictions_test.csv'}")
    print(f"Confusion matrix: {args.saida / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
