from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pipeline_utils import (
    CLASS_LABELS_PT,
    CLASS_NAMES,
    resolve_prediction_image,
    roc_curve_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera as figuras do TCC a partir dos arquivos salvos da CNN propria."
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "resultados_tcc",
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "dataset_cnn_propria"
        / "split_summary.csv",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "dataset_cnn_propria",
        help="Pasta do dataset dividido, usada para localizar as imagens de teste.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "figuras_tcc",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def generate_pipeline_figure(output_dir: Path) -> None:
    steps = [
        "Imagens tratadas\ncrop, inpainting e CLAHE",
        "Auditoria\nduplicatas e sequencias",
        "Divisao agrupada\ntreino, validacao e teste",
        "CNN propria\nresidual e multiescala",
        "Avaliacao\nmetricas e figuras",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    positions = np.linspace(0.08, 0.92, len(steps))
    for index, (x_position, label) in enumerate(zip(positions, steps)):
        ax.text(
            x_position,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#fff1df",
                "edgecolor": "#a65400",
                "linewidth": 1.2,
            },
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(positions[index + 1] - 0.09, 0.52),
                xytext=(x_position + 0.09, 0.52),
                arrowprops={"arrowstyle": "->", "color": "#7d4d20", "linewidth": 1.4},
            )
    ax.set_title("Fluxo metodologico da CNN propria", fontsize=14, pad=16)
    save_figure(output_dir / "figura_fluxo_cnn_propria.png")


def generate_split_distribution(summary_path: Path, output_dir: Path) -> None:
    summary = pd.read_csv(summary_path)
    summary = summary[summary["class"] != "total"].copy()
    summary["class_label"] = summary["class"].map(CLASS_LABELS_PT)
    summary["split"] = pd.Categorical(
        summary["split"], ["train", "val", "test"], ordered=True
    )
    pivot = summary.pivot(index="split", columns="class_label", values="count")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), color=["#e17c05", "#76b7b2"])
    ax.set_title("Distribuicao das imagens por subconjunto")
    ax.set_xlabel("Subconjunto")
    ax.set_ylabel("Numero de imagens")
    ax.set_xticklabels(["Treinamento", "Validacao", "Teste"], rotation=0)
    ax.legend(title="Classe")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fontsize=9)
    save_figure(output_dir / "figura_distribuicao_subconjuntos_cnn_propria.png")


def generate_training_curves(history_path: Path, output_dir: Path) -> None:
    history = pd.read_csv(history_path)
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, history["accuracy"], marker="o", label="Treino")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validacao")
    axes[0].set_title("Acuracia")

    axes[1].plot(epochs, history["loss"], marker="o", label="Treino")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validacao")
    axes[1].set_title("Perda")

    axes[2].plot(epochs, history["auc"], marker="o", label="Treino")
    axes[2].plot(epochs, history["val_auc"], marker="o", label="Validacao")
    axes[2].set_title("AUC")

    for axis in axes:
        axis.set_xlabel("Epoca")
        axis.set_ylabel("Valor")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Curvas de treinamento da CNN propria", fontsize=15)
    save_figure(output_dir / "figura_curvas_treinamento_cnn_propria.png")


def confusion_values(predictions: pd.DataFrame, threshold: float) -> np.ndarray:
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_prob = predictions["prob_endometriosis"].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def generate_confusion_matrix(
    predictions: pd.DataFrame,
    output_dir: Path,
    threshold: float,
) -> None:
    matrix = confusion_values(predictions, threshold)
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
    save_figure(output_dir / "figura_matriz_confusao_cnn_propria.png")


def generate_roc_curve(predictions: pd.DataFrame, output_dir: Path) -> None:
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_prob = predictions["prob_endometriosis"].astype(float).to_numpy()
    fpr, tpr, auc = roc_curve_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e17c05", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_title("Curva ROC - CNN propria")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(output_dir / "figura_curva_roc_cnn_propria.png")


def choose_error_examples(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    predicted_labels = (
        predictions["prob_endometriosis"].astype(float) >= threshold
    ).astype(int)
    errors = predictions[
        predictions["true_label"].astype(int) != predicted_labels
    ].copy()
    errors["figure_predicted_label"] = predicted_labels.loc[errors.index]
    false_positives = errors[errors["true_label"] == 0].sort_values(
        "prob_endometriosis", ascending=False
    )
    false_negatives = errors[errors["true_label"] == 1].sort_values(
        "prob_endometriosis", ascending=True
    )
    return pd.concat(
        [false_positives.head(3), false_negatives.head(3)], ignore_index=True
    )


def generate_error_examples(
    predictions: pd.DataFrame,
    output_dir: Path,
    dataset_dir: Path,
    threshold: float,
) -> None:
    selected = choose_error_examples(predictions, threshold)
    if selected.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for axis in axes.flat:
        axis.axis("off")

    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        image_path = resolve_prediction_image(
            row["file"],
            dataset_dir,
            str(row["true_class"]),
        )
        if image_path is None:
            continue
        image = Image.open(image_path).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        predicted_class = CLASS_NAMES[int(row["figure_predicted_label"])]
        axis.set_title(
            "Real: "
            f"{CLASS_LABELS_PT[row['true_class']]}\n"
            "Prevista: "
            f"{CLASS_LABELS_PT[predicted_class]}\n"
            f"Prob. endometriose: {float(row['prob_endometriosis']):.3f}",
            fontsize=9,
        )
    fig.suptitle("Exemplos de classificacoes incorretas - CNN propria", fontsize=14)
    save_figure(output_dir / "figura_exemplos_erros_cnn_propria.png")


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    predictions_path = args.resultados / "predictions_test.csv"
    history_path = args.resultados / "history.csv"
    for required_path in (predictions_path, history_path, args.split_summary):
        if not required_path.exists():
            raise FileNotFoundError(f"Arquivo necessario nao encontrado: {required_path}")

    predictions = pd.read_csv(predictions_path)
    generate_pipeline_figure(args.saida)
    generate_split_distribution(args.split_summary, args.saida)
    generate_training_curves(history_path, args.saida)
    generate_confusion_matrix(predictions, args.saida, args.threshold)
    generate_roc_curve(predictions, args.saida)
    generate_error_examples(predictions, args.saida, args.dataset, args.threshold)

    print(f"Figuras geradas em: {args.saida}")


if __name__ == "__main__":
    main()
