from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ("no_endometriosis", "endometriosis")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o melhor modelo YOLO-cls no conjunto de teste."
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados" / "dataset_yolo_cls",
        help="Pasta contendo train, val e test.",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("yolo11n_cls_experimento")
        / "resultados"
        / "resultados_tcc"
        / "treino_yolo_cls"
        / "weights"
        / "best.pt",
        help="Caminho do modelo treinado.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados" / "resultados_tcc",
        help="Pasta para salvar metricas e predicoes.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Tamanho de entrada.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Limiar de decisao.")
    parser.add_argument("--device", default=None, help="Dispositivo. Ex.: 0 para GPU ou cpu.")
    return parser.parse_args()


def list_test_images(test_dir: Path) -> tuple[list[Path], np.ndarray]:
    file_paths: list[Path] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {class_dir}")

        images = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda path: path.name.lower(),
        )
        file_paths.extend(images)
        labels.extend([class_index] * len(images))

    if not file_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {test_dir}")

    return file_paths, np.array(labels, dtype=int)


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
    file_paths: list[Path],
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
                    "file": str(file_path),
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
    ax.set_title("Matriz de confusao - YOLO-cls")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks([0, 1], labels=["Sem endometriose", "Com endometriose"])
    ax.set_yticks([0, 1], labels=["Sem endometriose", "Com endometriose"])

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def get_endometriosis_probabilities(model: YOLO, file_paths: list[Path], args: argparse.Namespace) -> np.ndarray:
    predict_kwargs = {
        "source": [str(path) for path in file_paths],
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    probabilities: list[float] = []

    for result in results:
        names = result.names
        class_to_index = {name: index for index, name in names.items()}
        if "endometriosis" not in class_to_index:
            raise ValueError(
                "Classe 'endometriosis' nao encontrada no modelo. "
                f"Classes encontradas: {names}"
            )
        endometriosis_index = class_to_index["endometriosis"]
        probs = result.probs.data.detach().cpu().numpy()
        probabilities.append(float(probs[endometriosis_index]))

    return np.array(probabilities, dtype=float)


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    if not args.modelo.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {args.modelo}")

    file_paths, y_true = list_test_images(args.dados / "test")
    model = YOLO(str(args.modelo))
    y_prob = get_endometriosis_probabilities(model, file_paths, args)
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

    print("Avaliacao finalizada.")
    print(json.dumps(metrics, indent=2))
    print(f"Predicoes: {args.saida / 'predictions_test.csv'}")
    print(f"Matriz de confusao: {args.saida / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
