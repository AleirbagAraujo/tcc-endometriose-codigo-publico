from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CLASS_LABELS = {
    "no_endometriosis": "Sem endometriose",
    "endometriosis": "Com endometriose",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera figuras para o TCC a partir do experimento YOLO-cls."
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados" / "resultados_tcc",
        help="Pasta contendo predicoes e metricas finais.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("yolo11n_cls_experimento")
        / "resultados"
        / "resultados_tcc"
        / "treino_yolo_cls",
        help="Pasta da execucao de treinamento da Ultralytics.",
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path("yolo11n_cls_experimento")
        / "resultados"
        / "dataset_yolo_cls"
        / "split_summary.csv",
        help="CSV com a distribuicao dos subconjuntos.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados" / "figuras_tcc",
        help="Pasta de saida das figuras.",
    )
    parser.add_argument("--threshold", type=float, default=0.50, help="Limiar de decisao.")
    return parser.parse_args()


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_pipeline_figure(output_dir: Path) -> None:
    steps = [
        "Imagens tratadas\ncrop, inpainting e CLAHE",
        "Divisao estratificada\n70% / 15% / 15%",
        "Dataset YOLO-cls\npastas train/val/test",
        "Treinamento\nmodelo *-cls",
        "Avaliacao\nmatriz e metricas",
    ]

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")

    x_positions = np.linspace(0.08, 0.92, len(steps))
    for index, (x, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#eef6e8",
                "edgecolor": "#4e7c45",
                "linewidth": 1.3,
            },
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.09, 0.55),
                xytext=(x + 0.09, 0.55),
                arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#4e7c45"},
            )

    ax.set_title("Fluxo metodologico da etapa YOLO-cls", fontsize=13, pad=16)
    save_current_figure(output_dir / "figura_fluxo_yolo_cls.png")


def generate_split_distribution(split_summary_path: Path, output_dir: Path) -> None:
    summary = pd.read_csv(split_summary_path)
    summary = summary[summary["class"] != "total"].copy()
    summary["class_label"] = summary["class"].map(CLASS_LABELS)
    summary["split"] = pd.Categorical(summary["split"], ["train", "val", "test"], ordered=True)
    summary = summary.sort_values(["split", "class_label"])

    pivot = summary.pivot(index="split", columns="class_label", values="count")
    ax = pivot.plot(kind="bar", figsize=(8, 4), color=["#59a14f", "#e15759"])
    ax.set_title("Distribuicao das imagens por subconjunto")
    ax.set_xlabel("Subconjunto")
    ax.set_ylabel("Numero de imagens")
    ax.set_xticklabels(["Treinamento", "Validacao", "Teste"], rotation=0)
    ax.legend(title="Classe")
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fontsize=9)

    save_current_figure(output_dir / "figura_distribuicao_subconjuntos_yolo_cls.png")


def load_training_results(run_dir: Path) -> pd.DataFrame | None:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None
    history = pd.read_csv(results_path)
    history.columns = [column.strip() for column in history.columns]
    return history


def first_existing_column(dataframe: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in dataframe.columns:
            return name
    return None


def generate_training_curves(run_dir: Path, output_dir: Path) -> None:
    history = load_training_results(run_dir)
    if history is None or history.empty:
        return

    epoch_col = first_existing_column(history, ["epoch", "Epoch"])
    x_values = history[epoch_col] + 1 if epoch_col else np.arange(1, len(history) + 1)

    train_loss_col = first_existing_column(history, ["train/loss", "loss"])
    val_loss_col = first_existing_column(history, ["val/loss"])
    top1_col = first_existing_column(history, ["metrics/accuracy_top1", "metrics/accuracy_top1(B)"])
    top5_col = first_existing_column(history, ["metrics/accuracy_top5", "metrics/accuracy_top5(B)"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if train_loss_col:
        axes[0].plot(x_values, history[train_loss_col], marker="o", label="Treino")
    if val_loss_col:
        axes[0].plot(x_values, history[val_loss_col], marker="o", label="Validacao")
    axes[0].set_title("Perda")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Valor")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if top1_col:
        axes[1].plot(x_values, history[top1_col], marker="o", label="Top-1")
    if top5_col:
        axes[1].plot(x_values, history[top5_col], marker="o", label="Top-5")
    axes[1].set_title("Acuracia de validacao")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Valor")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Curvas de treinamento do YOLO-cls", fontsize=14)
    save_current_figure(output_dir / "figura_curvas_treinamento_yolo_cls.png")


def compute_confusion_values(predictions: pd.DataFrame, threshold: float) -> dict[str, int]:
    true_label = predictions["true_label"].astype(int).to_numpy()
    pred_label = (predictions["prob_endometriosis"].astype(float).to_numpy() >= threshold).astype(int)

    return {
        "tn": int(np.sum((true_label == 0) & (pred_label == 0))),
        "fp": int(np.sum((true_label == 0) & (pred_label == 1))),
        "fn": int(np.sum((true_label == 1) & (pred_label == 0))),
        "tp": int(np.sum((true_label == 1) & (pred_label == 1))),
    }


def generate_confusion_matrix(predictions_path: Path, output_dir: Path, threshold: float) -> None:
    predictions = pd.read_csv(predictions_path)
    values = compute_confusion_values(predictions, threshold)
    matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(matrix, cmap="Greens")
    fig.colorbar(image, ax=ax)
    ax.set_title("Matriz de confusao - YOLO-cls")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks([0, 1], ["Sem endometriose", "Com endometriose"])
    ax.set_yticks([0, 1], ["Sem endometriose", "Com endometriose"])

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=13)

    save_current_figure(output_dir / "figura_matriz_confusao_yolo_cls.png")


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    tpr_values = []
    fpr_values = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr_values.append(tp / positives if positives else 0)
        fpr_values.append(fp / negatives if negatives else 0)

    fpr = np.array(fpr_values)
    tpr = np.array(tpr_values)
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    return fpr[order], tpr[order], auc


def generate_roc_curve(predictions_path: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_score = predictions["prob_endometriosis"].astype(float).to_numpy()
    fpr, tpr, auc = roc_curve_points(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#59a14f", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Classificador aleatorio")
    ax.set_title("Curva ROC - YOLO-cls")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.grid(alpha=0.25)
    ax.legend()
    save_current_figure(output_dir / "figura_curva_roc_yolo_cls.png")


def choose_error_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions[predictions["true_label"] != predictions["predicted_label"]].copy()
    false_positives = errors[errors["true_label"] == 0].sort_values(
        "prob_endometriosis", ascending=False
    )
    false_negatives = errors[errors["true_label"] == 1].sort_values(
        "prob_endometriosis", ascending=True
    )
    return pd.concat([false_positives.head(4), false_negatives.head(2)], ignore_index=True)


def generate_error_examples(predictions_path: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    selected = choose_error_examples(predictions)

    if selected.empty:
        return

    columns = 3
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, selected.iterrows()):
        image_path = Path(row["file"])
        if not image_path.exists():
            image_path = Path(str(row["file"]).replace("\\", "/"))

        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        true_name = CLASS_LABELS[row["true_class"]]
        predicted_name = CLASS_LABELS[row["predicted_class"]]
        probability = float(row["prob_endometriosis"])
        ax.set_title(
            f"Real: {true_name}\nPrevista: {predicted_name}\nProb.: {probability:.3f}",
            fontsize=9,
        )

    fig.suptitle("Exemplos de classificacoes incorretas - YOLO-cls", fontsize=14)
    save_current_figure(output_dir / "figura_exemplos_erros_yolo_cls.png")


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    predictions_path = args.resultados / "predictions_test.csv"

    generate_pipeline_figure(args.saida)
    generate_split_distribution(args.split_summary, args.saida)
    generate_training_curves(args.run_dir, args.saida)
    generate_confusion_matrix(predictions_path, args.saida, args.threshold)
    generate_roc_curve(predictions_path, args.saida)
    generate_error_examples(predictions_path, args.saida)

    print("Figuras geradas em:")
    print(args.saida)


if __name__ == "__main__":
    main()
