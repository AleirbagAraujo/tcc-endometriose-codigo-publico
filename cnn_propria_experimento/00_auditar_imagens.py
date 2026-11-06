from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_utils import CLASS_NAMES, deduplicate_records, list_image_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita classes, grupos de sequencia e duplicatas exatas."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("tcc_fotos") / "dataset_tratado",
        help="Pasta das imagens tratadas organizadas por classe.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("cnn_propria_experimento")
        / "resultados"
        / "auditoria_imagens.json",
        help="Arquivo JSON de auditoria.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=25,
        help="Quantidade de indices consecutivos por grupo de sequencia.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list_image_records(args.origem, group_size=args.group_size)
    unique_records, duplicates = deduplicate_records(records)

    class_counts = Counter(record.class_name for record in records)
    unique_counts = Counter(record.class_name for record in unique_records)
    group_counts = Counter(
        (record.class_name, record.group_id) for record in unique_records
    )
    report = {
        "source": str(args.origem),
        "class_order": list(CLASS_NAMES),
        "positive_class": "endometriosis",
        "group_size": args.group_size,
        "total_images": len(records),
        "unique_images": len(unique_records),
        "exact_duplicates": len(duplicates),
        "classes": {
            class_name: {
                "total": class_counts[class_name],
                "unique": unique_counts[class_name],
                "groups": sum(1 for name, _ in group_counts if name == class_name),
            }
            for class_name in CLASS_NAMES
        },
        "duplicates": [
            {
                "class": canonical.class_name,
                "sha256": canonical.sha256,
                "kept": str(canonical.path),
                "removed": str(duplicate.path),
            }
            for canonical, duplicate in duplicates
        ],
    }

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAuditoria salva em: {args.saida}")


if __name__ == "__main__":
    main()
