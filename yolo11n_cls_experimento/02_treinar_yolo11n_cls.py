from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# Pasta em que este arquivo .py está localizado
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina o modelo YOLO11n-cls para classificacao "
            "binaria de imagens laparoscopicas."
        )
    )

    parser.add_argument(
        "--dados",
        type=Path,
        default=BASE_DIR / "resultados" / "dataset_yolo_cls",
        help="Pasta contendo train, val e test.",
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=BASE_DIR / "resultados" / "resultados_tcc",
        help="Pasta principal para salvar os resultados.",
    )

    parser.add_argument(
        "--nome-run",
        default="treino_yolo_cls",
        help="Nome da pasta criada para esta execucao.",
    )

    parser.add_argument(
        "--modelo",
        default="yolo11n-cls.pt",
        help="Checkpoint de classificacao utilizado no treinamento.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Dimensao das imagens de entrada.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Quantidade de imagens por lote.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero maximo de epocas.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Paciencia utilizada pelo early stopping.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatoria.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo de treinamento. Exemplo: 0 ou cpu.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Quantidade de workers do carregador de dados.",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado inicial.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout da cabeca classificadora.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Converte os caminhos para caminhos absolutos
    args.dados = args.dados.expanduser().resolve()
    args.saida = args.saida.expanduser().resolve()

    if not args.dados.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado: {args.dados}"
        )

    train_dir = args.dados / "train"
    val_dir = args.dados / "val"
    test_dir = args.dados / "test"

    for directory in (train_dir, val_dir, test_dir):
        if not directory.exists():
            raise FileNotFoundError(
                f"Divisao do dataset nao encontrada: {directory}"
            )

    args.saida.mkdir(parents=True, exist_ok=True)

    config = {
        "modelo": args.modelo,
        "dados": str(args.dados),
        "saida": str(args.saida),
        "nome_run": args.nome_run,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "lr0": args.lr0,
        "dropout": args.dropout,
    }

    config_path = args.saida / "config_treinamento.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Configuracao do treinamento")
    print(f"Modelo: {args.modelo}")
    print(f"Dataset: {args.dados}")
    print(f"Saida: {args.saida}")
    print(f"Epocas: {args.epochs}")
    print(f"Tamanho das imagens: {args.img_size} x {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print()

    modelo = YOLO(args.modelo)

    train_kwargs = {
        "data": str(args.dados),
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "project": str(args.saida),
        "name": args.nome_run,
        "exist_ok": True,
        "seed": args.seed,
        "patience": args.patience,
        "workers": args.workers,
        "lr0": args.lr0,
        "dropout": args.dropout,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    modelo.train(**train_kwargs)

    # Consulta diretamente o caminho utilizado pela Ultralytics
    run_dir = Path(modelo.trainer.save_dir).resolve()
    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"

    if not best_model.exists():
        raise FileNotFoundError(
            f"O treinamento terminou, mas best.pt nao foi encontrado: "
            f"{best_model}"
        )

    print()
    print("Treinamento finalizado.")
    print(f"Pasta real da execucao: {run_dir}")
    print(f"Melhor modelo: {best_model}")
    print(f"Ultimo modelo: {last_model}")
    print(f"Configuracao: {config_path}")


if __name__ == "__main__":
    main()