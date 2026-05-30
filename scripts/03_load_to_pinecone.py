# scripts/03_load_to_pinecone.py
# Execution: python scripts/03_load_to_pinecone.py

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Configuration constants
INPUT_PARQUET = Path("data/arxiv_subset.parquet")
INPUT_EMBEDDINGS = Path("embeddings/embeddings.npy")
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200  # Optimal batch size for stable network ingestion


def main() -> None:
    # 1. Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ПОМИЛКА] PINECONE_API_KEY не знайдено у змінних середовища або файлі .env")
        return

    pc = Pinecone(api_key=api_key)

    # 2. Create index if it does not exist
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося підключитися до Pinecone API: {e}")
        return

    if INDEX_NAME not in existing_indexes:
        print(f"[СТАТУС] Створення нового індексу '{INDEX_NAME}'...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",  # Using cosine similarity since embeddings are normalized
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Default region available in the free tier
                )
            )

            # Wait until the index is fully initialized and ready
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
            print("[УСПІХ] Індекс успішно створено та ініціалізовано.")
        except Exception as e:
            print(f"[ПОМИЛКА] Не вдалося створити індекс: {e}")
            return
    else:
        print(f"[ІНФО] Індекс '{INDEX_NAME}' вже існує.")

    # Connect to the target index
    index = pc.Index(INDEX_NAME)

    # 3. Load local dataset and pre-computed embeddings
    if not INPUT_PARQUET.exists() or not INPUT_EMBEDDINGS.exists():
        print("[ПОМИЛКА] Відсутні необхідні вхідні файли (Parquet або ембеддинги)!")
        return

    print("[СТАТУС] Завантаження локальних файлів з диска...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    if len(df) != len(embeddings):
        print(f"[ПОМИЛКА] Розбіжність розмірів: даних {len(df)}, ембеддингів {len(embeddings)}!")
        return

    # Convert DataFrame to a list of dictionaries for faster iteration than .iloc
    records = df.to_dict(orient="records")
    total_records = len(records)

    # 4. Prepare and upsert data in batches
    print(f"[СТАТУС] Запуск завантаження {total_records} векторів у Pinecone батчами по {BATCH_SIZE}...")

    for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Upserting to Pinecone"):
        batch_end = min(i + BATCH_SIZE, total_records)
        upsert_data = []

        for idx in range(i, batch_end):
            row = records[idx]

            # Construct metadata payload with safe character limits
            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],  # Bound to 500 characters
                "authors": str(row["authors"])[:200],    # Bound to 200 characters
                "year": int(row["year"]),
                "category": str(row["category"])
            }

            upsert_data.append({
                "id": f"paper_{idx}",
                "values": embeddings[idx].tolist(),  # Native Python list format required by Pinecone
                "metadata": metadata
            })

        # Submit the prepared batch to Pinecone
        try:
            index.upsert(vectors=upsert_data)
        except Exception as e:
            print(f"\n[ПОМИЛКА] Не вдалося відправити батч починаючи з індексу {i}: {e}")
            return

    # 5. Verify index state
    print("\n[УСПІХ] Завантаження даних завершено повністю.")
    print("[СТАТУС] Очікування оновлення індексу для перевірки статистики...")
    time.sleep(2)  # Short block to allow Pinecone to refresh index metrics

    try:
        stats = index.describe_index_stats()
        print("-" * 45)
        print(f"Загальна кількість векторів в індексі: {stats['total_vector_count']}")
        print("-" * 45)
    except Exception as e:
        print(f"[ПОПЕРЕДЖЕННЯ] Не вдалося отримати фінальну статистику: {e}")


if __name__ == "__main__":
    main()