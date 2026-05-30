# scripts/02_embed.py
# Execution: python scripts/02_embed.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configuration constants
INPUT_FILE = Path("data/arxiv_subset.parquet")
OUTPUT_DIR = Path("embeddings")
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def print_stats(embeddings: np.ndarray) -> None:
    """Calculate and display basic embedding metrics."""
    embedding_dim = embeddings.shape[1]
    first_norm = np.linalg.norm(embeddings[0])

    print("\n" + "=" * 40)
    print(f"{'СТАТИСТИКА ЕМБЕДДИНГІВ':^40}")
    print("=" * 40)
    print(f" Загальна кількість: {len(embeddings):<20}")
    print(f" Розмірність:        {embedding_dim:<20}")
    print(f" Норма першого вектора: {first_norm:<17.4f}")
    print("=" * 40 + "\n")


def main() -> None:
    # 1. Load dataset
    if not INPUT_FILE.exists():
        print(f"[ПОМИЛКА] Вхідний файл '{INPUT_FILE}' не знайдено!")
        return

    print(f"[ІНФО] Завантаження даних з {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося прочитати Parquet-файл: {e}")
        return

    num_records = len(df)
    print(f"[УСПІХ] Завантажено записів: {num_records}")

    # 1.1 Validate existing embeddings
    if OUTPUT_FILE.exists():
        print(f"[СТАТУС] Виявлено існуючий файл {OUTPUT_FILE}. Перевірка валідності...")
        try:
            embeddings = np.load(OUTPUT_FILE)
            if len(embeddings) == num_records:
                print(f"[ІНФО] Знайдено {len(embeddings)} валідних ембеддингів. Генерація пропускається.")
                print_stats(embeddings)
                return
            else:
                print(f"[ПОПЕРЕДЖЕННЯ] Невідповідність розмірів (у файлі: {len(embeddings)}, у датасеті: {num_records}).")
        except Exception as e:
            print(f"[ПОПЕРЕДЖЕННЯ] Помилка при читанні існуючих ембеддингів: {e}")

        print("[СТАТУС] Буде запущено повторну генерацію ембеддингів.")

    # 2. Prepare texts using the format expected by specter2
    print("[СТАТУС] Підготовка текстів для кодування (формат: title + [SEP] + abstract)...")
    texts = (df['title'] + " [SEP] " + df['abstract']).tolist()

    # 3. Initialize model and select processing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ІНФО] Завантаження моделі '{MODEL_NAME}' на пристрій: [{device.upper()}]...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Generate embeddings
    print(f"[СТАТУС] Обчислення ембеддингів (batch_size={BATCH_SIZE}). Будь ласка, зачекайте...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True  # Ensures embeddings have unit length (norm ~1.0)
    )

    # 5. Display processing stats
    print_stats(embeddings)

    # 6. Save results to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"[УСПІХ] Ембеддинги успішно збережені у: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()