from __future__ import annotations

import argparse
import csv
from pathlib import Path


CLASSES = ("no_endometriosis", "endometriosis")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confere a base de imagens tratadas que sera usada no experimento YOLO-cls."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("tcc_fotos") / "dataset_tratado",
        help="Pasta gerada pelo script 00 de tratamento das imagens.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("yolo11n_cls_experimento") / "resultados",
        help="Pasta em que o resumo da conferencia sera salvo.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.origem.exists():
        raise FileNotFoundError(
            f"Pasta de imagens tratadas nao encontrada: {args.origem}. "
            "Execute antes o tratamento das imagens."
        )

    summary: list[dict[str, object]] = []
    total = 0

    for class_name in CLASSES:
        class_dir = args.origem / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {class_dir}")

        images = list_images(class_dir)
        if not images:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {class_dir}")

        count = len(images)
        total += count
        summary.append({"class": class_name, "count": count})

    summary.append({"class": "total", "count": total})
    write_csv(args.saida / "dataset_tratado_summary.csv", summary)

    print("Base tratada conferida com sucesso.")
    print(f"Origem: {args.origem}")
    for row in summary:
        print(f"{row['class']:<16} | {row['count']}")


if __name__ == "__main__":
    main()
