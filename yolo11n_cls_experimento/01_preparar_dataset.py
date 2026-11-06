from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


CLASSES = ("no_endometriosis", "endometriosis")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide a base tratada em train, val e test para classificacao com YOLO."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("tcc_fotos") / "dataset_tratado",
        help="Pasta com as imagens tratadas divididas por classe.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados" / "dataset_yolo_cls",
        help="Pasta de saida no formato de classificacao esperado pela Ultralytics.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Proporcao de treino.")
    parser.add_argument("--val", type=float, default=0.15, help="Proporcao de validacao.")
    parser.add_argument("--test", type=float, default=0.15, help="Proporcao de teste.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recria a pasta de saida caso ela ja exista.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"As proporcoes devem somar 1. Soma atual: {total:.6f}")


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def split_class_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_output_folder(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Pasta de saida ja existe: {output_dir}. Use --overwrite para recriar."
            )
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_split_files(
    output_dir: Path,
    split_map: dict[str, dict[str, list[Path]]],
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []

    for class_name, splits in split_map.items():
        for split_name, files in splits.items():
            destination_dir = output_dir / split_name / class_name
            for source_path in files:
                destination_path = destination_dir / source_path.name
                shutil.copy2(source_path, destination_path)
                manifest.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "source": str(source_path),
                        "destination": str(destination_path),
                    }
                )

    return manifest


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(manifest: list[dict[str, str]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_total = 0
        for class_name in CLASSES:
            count = sum(
                1
                for row in manifest
                if row["split"] == split and row["class"] == class_name
            )
            split_total += count
            summary_rows.append({"split": split, "class": class_name, "count": count})
        summary_rows.append({"split": split, "class": "total", "count": split_total})
    return summary_rows


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    if not args.origem.exists():
        raise FileNotFoundError(f"Pasta de origem nao encontrada: {args.origem}")

    split_map: dict[str, dict[str, list[Path]]] = {}
    for class_name in CLASSES:
        class_dir = args.origem / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {class_dir}")

        files = list_images(class_dir)
        if not files:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {class_dir}")

        split_map[class_name] = split_class_files(
            files=files,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
        )

    prepare_output_folder(args.saida, args.overwrite)
    manifest = copy_split_files(args.saida, split_map)
    summary = build_summary(manifest)

    write_csv(
        args.saida / "split_manifest.csv",
        manifest,
        ["split", "class", "source", "destination"],
    )
    write_csv(args.saida / "split_summary.csv", summary, ["split", "class", "count"])

    print("Dataset YOLO-cls criado com sucesso.")
    print(f"Origem: {args.origem}")
    print(f"Saida: {args.saida}")
    print()
    for row in summary:
        print(f"{row['split']:>5} | {row['class']:<16} | {row['count']}")


if __name__ == "__main__":
    main()
