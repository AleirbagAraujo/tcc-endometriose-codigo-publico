from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from pipeline_utils import (
    CLASS_NAMES,
    SPLIT_NAMES,
    ImageRecord,
    deduplicate_records,
    list_image_records,
    split_grouped_records,
    validate_removable_output,
    validate_ratios,
)


RESULTS_ROOT = Path("cnn_propria_experimento") / "resultados"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria train, val e test sem duplicatas e agrupando quadros consecutivos."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("tcc_fotos") / "dataset_tratado",
        help="Pasta das imagens tratadas divididas por classe.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "dataset_cnn_propria",
        help="Pasta de saida do novo conjunto.",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-size",
        type=int,
        default=25,
        help="Quantidade de indices consecutivos mantidos no mesmo grupo.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_folder(
    output_dir: Path,
    overwrite: bool,
    allowed_root: Path = RESULTS_ROOT,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Pasta de saida ja existe: {output_dir}. Use --overwrite para recriar."
            )
        resolved = validate_removable_output(output_dir, allowed_root)
        shutil.rmtree(resolved)

    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_split_map(
    split_map: dict[str, list[ImageRecord]],
) -> None:
    hash_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for split_name, records in split_map.items():
        for record in records:
            previous_hash = hash_owners.setdefault(record.sha256, split_name)
            if previous_hash != split_name:
                raise RuntimeError(f"Duplicata atravessou subconjuntos: {record.path}")
            previous_group = group_owners.setdefault(record.group_id, split_name)
            if previous_group != split_name:
                raise RuntimeError(
                    f"Grupo atravessou subconjuntos: {record.group_id}"
                )


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    records = list_image_records(args.origem, group_size=args.group_size)
    unique_records, duplicates = deduplicate_records(records)
    prepare_output_folder(args.saida, args.overwrite)

    split_map = split_grouped_records(
        unique_records,
        args.train,
        args.val,
        args.test,
        seed=args.seed,
    )

    validate_split_map(split_map)
    manifest: list[dict[str, object]] = []
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            class_records = [
                record
                for record in split_map[split_name]
                if record.class_name == class_name
            ]
            for record in class_records:
                destination = args.saida / split_name / class_name / record.path.name
                shutil.copy2(record.path, destination)
                manifest.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "group_id": record.group_id,
                        "sha256": record.sha256,
                        "source": str(record.path),
                        "destination": str(destination),
                    }
                )

    summary: list[dict[str, object]] = []
    for split_name in SPLIT_NAMES:
        split_total = 0
        for class_name in CLASS_NAMES:
            count = sum(
                1
                for row in manifest
                if row["split"] == split_name and row["class"] == class_name
            )
            split_total += count
            summary.append({"split": split_name, "class": class_name, "count": count})
        summary.append({"split": split_name, "class": "total", "count": split_total})

    duplicate_rows = [
        {
            "class": canonical.class_name,
            "sha256": canonical.sha256,
            "kept": str(canonical.path),
            "removed": str(duplicate.path),
        }
        for canonical, duplicate in duplicates
    ]
    write_csv(
        args.saida / "split_manifest.csv",
        manifest,
        ["split", "class", "group_id", "sha256", "source", "destination"],
    )
    write_csv(
        args.saida / "split_summary.csv",
        summary,
        ["split", "class", "count"],
    )
    write_csv(
        args.saida / "duplicates_removed.csv",
        duplicate_rows,
        ["class", "sha256", "kept", "removed"],
    )

    config = {
        "source": str(args.origem),
        "output": str(args.saida),
        "class_order": list(CLASS_NAMES),
        "positive_class": "endometriosis",
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
        "group_size": args.group_size,
        "grouping_strategy": "filename_prefix_and_consecutive_index_block_proxy",
        "grouping_limitation": (
            "Use patient, surgery or video identifiers when they become available."
        ),
        "original_images": len(records),
        "unique_images": len(unique_records),
        "duplicates_removed": len(duplicates),
        "unique_images_by_class": dict(
            Counter(record.class_name for record in unique_records)
        ),
        "groups_by_class": {
            class_name: len(
                {
                    record.group_id
                    for record in unique_records
                    if record.class_name == class_name
                }
            )
            for class_name in CLASS_NAMES
        },
    }
    with (args.saida / "split_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    print("Dataset da CNN propria criado com sucesso.")
    print(f"Imagens originais: {len(records)}")
    print(f"Duplicatas removidas: {len(duplicates)}")
    print(f"Imagens utilizadas: {len(unique_records)}")
    for row in summary:
        print(f"{row['split']:>5} | {row['class']:<16} | {row['count']}")
    print(f"Saida: {args.saida}")


if __name__ == "__main__":
    main()
