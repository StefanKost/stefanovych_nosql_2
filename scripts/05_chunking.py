# scripts/05_chunking.py
# Execution: python scripts/05_chunking.py

import os
import re
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration constants
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
BATCH_SIZE = 100
DATA_PATH = Path("data/arxiv_subset.parquet")


def get_fixed_chunks(text: str, size: int = 100, overlap: int = 20) -> list:
    """Split text into fixed-size word tokens with a predefined sliding overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def get_semantic_chunks(text: str, max_words: int = 100) -> list:
    """Group complete sentences without breaking them until reaching the maximum word threshold."""
    # Basic sentence boundary splitting using regex
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_count = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_count + sent_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
        current_chunk.append(sent)
        current_count += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_and_upload(index_name: str, strategy_func, df_papers: pd.DataFrame, pc_client: Pinecone, model_instance: SentenceTransformer) -> None:
    """Process document chunking according to the strategy, embed text structures and upload to Pinecone."""
    index = pc_client.Index(index_name)
    all_vectors = []

    print(f"\n[СТАТУС] Обробка та завантаження для індексу: {index_name}...")

    for _, row in tqdm(df_papers.iterrows(), total=len(df_papers), desc=f"Processing {index_name}"):
        chunks = strategy_func(row['abstract'])

        # Specter2 expects the format: Title + [SEP] + Content chunk
        texts_to_embed = [f"{row['title']} [SEP] {c}" for c in chunks]
        embeddings = model_instance.encode(texts_to_embed, normalize_embeddings=True)

        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            all_vectors.append({
                "id": f"{row['id']}_ch{i}_{index_name}",
                "values": emb.tolist(),
                "metadata": {
                    "arxiv_id": str(row['id']),
                    "title": str(row['title']),
                    "text": chunk_text[:1000],  # Bound metadata string capacity constraints
                    "chunk_num": int(i),
                    "year": int(row['year']),
                    "category": str(row['category'])
                }
            })

            # Submit batch immediately upon filling buffer capacity
            if len(all_vectors) >= BATCH_SIZE:
                try:
                    index.upsert(vectors=all_vectors)
                except Exception as e:
                    print(f"\n[ПОМИЛКА] Не вдалося відправити батч до {index_name}: {e}")
                all_vectors = []

    # Flush remaining records left within the batch container array
    if all_vectors:
        try:
            index.upsert(vectors=all_vectors)
        except Exception as e:
            print(f"\n[ПОМИЛКА] Не вдалося відправити фінальний батч до {index_name}: {e}")


def search_demo(query: str, pc_client: Pinecone, model_instance: SentenceTransformer) -> None:
    """Execute evaluation queries against both chunked indices to showcase difference in rankings."""
    print("\n" + "=" * 60)
    print(f" ДЕМОНСТРАЦІЯ ПОШУКУ ЗА ЗАПИТОМ: '{query}'")
    print("=" * 60)

    v = model_instance.encode(query, normalize_embeddings=True).tolist()

    for name in [FIXED_INDEX, SEMANTIC_INDEX]:
        print(f"\n--- Результати пошуку з індексу: {name} ---")
        try:
            res = pc_client.Index(name).query(vector=v, top_k=3, include_metadata=True)
            for m in res['matches']:
                score = m.get('score')
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                print(f" Score (Схожість): {score_str} | Назва: {m['metadata']['title'][:60]}...")
                print(f" Текст чанка: {m['metadata']['text'][:140]}...\n")
        except Exception as e:
            print(f"[ПОМИЛКА] Не вдалося виконати пошуковий запит для {name}: {e}")


def main() -> None:
    # Verify environment initialization parameters
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ПОМИЛКА] PINECONE_API_KEY не знайдено у змінних середовища або файлі .env")
        return

    if not DATA_PATH.exists():
        print(f"[ПОМИЛКА] Не знайдено вхідний файл даних за шляхом: {DATA_PATH}")
        return

    print("[СТАТУС] Ініціалізація моделей та підключення до Pinecone...")
    pc = Pinecone(api_key=api_key)
    model = SentenceTransformer(MODEL_NAME)

    # 1. Prepare target corpus dataset (Top 30 longest abstracts)
    df = pd.read_parquet(DATA_PATH)
    df['abstract_len'] = df['abstract'].str.len()
    top_30_papers = df.nlargest(30, 'abstract_len').copy()
    print(f"[ІНФО] Відібрано {len(top_30_papers)} найдовших анотацій для аналізу чанкінгу.")

    # 2. Infrastructure generation loops
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося підключитися до Pinecone API: {e}")
        return

    for name in [FIXED_INDEX, SEMANTIC_INDEX]:
        if name not in existing_indexes:
            print(f"[СТАТУС] Створення індексу '{name}'...")
            try:
                pc.create_index(
                    name=name, dimension=VECTOR_DIM, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                while not pc.describe_index(name).status['ready']:
                    time.sleep(1)
                print(f"[УСПІХ] Індекс '{name}' готовий до використання.")
            except Exception as e:
                print(f"[ПОМИЛКА] Не вдалося створити індекс {name}: {e}")
                return

    # 3. Process partitioning routes
    process_and_upload(FIXED_INDEX, get_fixed_chunks, top_30_papers, pc, model)
    process_and_upload(SEMANTIC_INDEX, get_semantic_chunks, top_30_papers, pc, model)

    # 4. Trigger analytical test queries
    search_demo("experimental verification of neural network stability", pc, model)


if __name__ == "__main__":
    main()