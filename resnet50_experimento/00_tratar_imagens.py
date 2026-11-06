import argparse
from pathlib import Path
import cv2
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica pré-processamento (Inpainting, CLAHE e Crop) nas imagens laparoscópicas."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("tcc_fotos") / "dataset_final",
        help="Pasta raiz com as imagens originais divididas por classe.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("tcc_fotos") / "dataset_tratado",
        help="Pasta de saída para as imagens processadas.",
    )
    return parser.parse_args()

def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Remove as bordas pretas escuras ao redor da imagem laparoscópica."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img

def remove_specular_reflections(img: np.ndarray) -> np.ndarray:
    """Detecta reflexos brancos do laparoscópio e os preenche (Inpainting)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Limiar alto para encontrar os pixels estourados (brancos)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # Dilatar levemente a máscara para pegar as bordas do reflexo
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Aplica o algoritmo de Inpainting de Telea
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Aplica equalização de histograma adaptativa para melhorar o contraste das lesões."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplica CLAHE no canal de Luminosidade (L)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def process_image(src_path: Path, dest_path: Path) -> None:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"Erro ao ler imagem: {src_path}")
        return

    # Pipeline de tratamento em 3 etapas
    img = crop_black_borders(img)
    img = remove_specular_reflections(img)
    img = apply_clahe(img)

    cv2.imwrite(str(dest_path), img)

def main() -> None:
    args = parse_args()
    
    if not args.origem.exists():
        raise FileNotFoundError(f"Pasta de origem não encontrada: {args.origem}")

    # Cria a estrutura de saída baseada na origem
    classes = [d.name for d in args.origem.iterdir() if d.is_dir()]
    total_images = 0

    for class_name in classes:
        src_class_dir = args.origem / class_name
        dest_class_dir = args.saida / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = [f for f in src_class_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        print(f"Processando {len(images)} imagens da classe '{class_name}'...")
        
        for img_path in images:
            dest_path = dest_class_dir / img_path.name
            process_image(img_path, dest_path)
            total_images += 1

    print(f"\nTratamento concluído com sucesso! Total: {total_images} imagens.")
    print(f"Imagens salvas em: {args.saida}")

if __name__ == "__main__":
    main()