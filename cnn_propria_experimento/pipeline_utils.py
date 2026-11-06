from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ("no_endometriosis", "endometriosis")
CLASS_LABELS_PT = {
    "no_endometriosis": "Sem endometriose",
    "endometriosis": "Com endometriose",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    class_name: str
    path: Path
    sha256: str
    group_id: str


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("As proporcoes devem ser maiores que zero.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"As proporcoes devem somar 1. Soma atual: {sum(ratios):.6f}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_group_id(path: Path, group_size: int) -> str:
    if group_size <= 0:
        raise ValueError("group_size deve ser maior que zero.")

    match = re.match(r"^(.*?)[_-](\d+)$", path.stem)
    if not match:
        return path.stem.lower()

    prefix, numeric_text = match.groups()
    block = int(numeric_text) // group_size
    return f"{prefix.lower()}:{block}"


def list_image_records(source_dir: Path, group_size: int = 25) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {class_dir}")

        images = sorted(
            (
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        if not images:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {class_dir}")

        for path in images:
            records.append(
                ImageRecord(
                    class_name=class_name,
                    path=path,
                    sha256=sha256_file(path),
                    group_id=infer_group_id(path, group_size),
                )
            )
    return records


def deduplicate_records(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[tuple[ImageRecord, ImageRecord]]]:
    canonical_by_hash: dict[str, ImageRecord] = {}
    duplicates: list[tuple[ImageRecord, ImageRecord]] = []

    for record in sorted(records, key=lambda item: str(item.path).lower()):
        canonical = canonical_by_hash.get(record.sha256)
        if canonical is None:
            canonical_by_hash[record.sha256] = record
            continue
        if canonical.class_name != record.class_name:
            raise ValueError(
                "Imagem identica encontrada em classes diferentes: "
                f"{canonical.path} e {record.path}"
            )
        duplicates.append((canonical, record))

    return list(canonical_by_hash.values()), duplicates


def validate_removable_output(output_dir: Path, allowed_root: Path) -> Path:
    resolved_output = output_dir.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_output == resolved_root or not resolved_output.is_relative_to(
        resolved_root
    ):
        raise ValueError(
            "Pasta de saida insegura para remocao. "
            f"Ela deve estar dentro de {resolved_root}: {resolved_output}"
        )
    return resolved_output


def split_grouped_records(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    if not records:
        raise ValueError("Nenhuma imagem disponivel para divisao.")

    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)

    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("Sao necessarios ao menos tres grupos para criar train, val e test.")

    rng = random.Random(seed)
    grouped_items = list(groups.items())
    rng.shuffle(grouped_items)
    grouped_items.sort(key=lambda item: len(item[1]), reverse=True)

    ratios = dict(zip(SPLIT_NAMES, (train_ratio, val_ratio, test_ratio)))
    total_targets = {name: len(records) * ratio for name, ratio in ratios.items()}
    class_totals = {
        class_name: sum(record.class_name == class_name for record in records)
        for class_name in CLASS_NAMES
    }
    class_targets = {
        split_name: {
            class_name: class_totals[class_name] * ratios[split_name]
            for class_name in CLASS_NAMES
        }
        for split_name in SPLIT_NAMES
    }
    split_map: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    split_class_counts = {
        split_name: {class_name: 0 for class_name in CLASS_NAMES}
        for split_name in SPLIT_NAMES
    }

    def allocation_score(
        candidate: str,
        group_records: list[ImageRecord],
    ) -> tuple[float, int, int]:
        group_class_counts = {
            class_name: sum(
                record.class_name == class_name for record in group_records
            )
            for class_name in CLASS_NAMES
        }
        score = 0.0
        for split_name in SPLIT_NAMES:
            total_count = len(split_map[split_name])
            if split_name == candidate:
                total_count += len(group_records)
            total_target = max(total_targets[split_name], 1.0)
            score += ((total_count - total_target) / total_target) ** 2

            for class_name in CLASS_NAMES:
                class_count = split_class_counts[split_name][class_name]
                if split_name == candidate:
                    class_count += group_class_counts[class_name]
                class_target = max(class_targets[split_name][class_name], 1.0)
                score += ((class_count - class_target) / class_target) ** 2

        return score, len(split_map[candidate]), SPLIT_NAMES.index(candidate)

    for item_index, (_, group_records) in enumerate(grouped_items):
        empty_splits = [name for name in SPLIT_NAMES if not split_map[name]]
        remaining_groups = len(grouped_items) - item_index
        candidates = (
            empty_splits
            if empty_splits and remaining_groups == len(empty_splits)
            else list(SPLIT_NAMES)
        )
        split_name = min(
            candidates,
            key=lambda name: allocation_score(name, group_records),
        )
        split_map[split_name].extend(group_records)
        for record in group_records:
            split_class_counts[split_name][record.class_name] += 1

    for split_records in split_map.values():
        split_records.sort(key=lambda record: str(record.path).lower())
    return split_map


def resolve_prediction_image(
    file_path: str | Path,
    dataset_dir: Path,
    true_class: str,
) -> Path | None:
    """Localiza uma imagem mesmo quando o CSV guarda um caminho antigo do Colab."""
    original_path = Path(file_path)
    if original_path.exists():
        return original_path

    if true_class not in CLASS_NAMES:
        raise ValueError(f"Classe desconhecida no arquivo de predicoes: {true_class}")

    local_path = dataset_dir / "test" / true_class / original_path.name
    return local_path if local_path.exists() else None


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


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

    index = 0
    while index < len(scores):
        last = index
        while last + 1 < len(scores) and scores[order[last + 1]] == scores[order[index]]:
            last += 1
        ranks[order[index : last + 1]] = (index + last + 2) / 2.0
        index = last + 1

    positive_ranks = ranks[:n_pos]
    return float(
        (positive_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def roc_curve_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y_true) != len(y_score):
        raise ValueError("y_true e y_score devem possuir o mesmo tamanho.")

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("A curva ROC exige exemplos das duas classes.")

    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    fpr_values: list[float] = []
    tpr_values: list[float] = []
    for threshold in thresholds:
        predicted = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (predicted == 1)))
        fp = int(np.sum((y_true == 0) & (predicted == 1)))
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)

    fpr = np.asarray(fpr_values)
    tpr = np.asarray(tpr_values)
    order = np.lexsort((tpr, fpr))
    return fpr[order], tpr[order], binary_auc(y_true, y_score)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true e y_prob devem possuir o mesmo tamanho.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "positive_class": "endometriosis",
        "total": int(len(y_true)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": safe_divide(tp + tn, len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": safe_divide(2 * precision * recall, precision + recall),
        "auc": binary_auc(y_true, y_prob),
    }
