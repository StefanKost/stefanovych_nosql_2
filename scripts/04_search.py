# scripts/04_search.py
# Execution: python scripts/04_search.py

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# Load environment variables from .env file
load_dotenv()

# Configuration constants
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5


def encode_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Encode the text query using the sentence transformer model with normalization."""
    return model.encode(query, normalize_embeddings=True)


def semantic_search(query: str, index: Pinecone.Index, model: SentenceTransformer, top_k: int = TOP_K, filters: dict = None) -> None:
    """Perform a vector similarity search on the remote Pinecone index with optional metadata filtering."""
    query_vec = encode_query(query, model).tolist()

    try:
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=filters)
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося виконати запит до Pinecone: {e}")
        return

    print("\n" + "=" * 75)
    print(f" ПОШУК У PINECONE: {query}")
    if filters:
        print(f" Фільтрація: {filters}")
    print("=" * 75)

    for match in results["matches"]:
        md = match.get("metadata", {})
        score = match.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

        print(f"- {md.get('title')} ({md.get('category')}, {md.get('year')})")
        print(f"  Схожість (Score): {score_str}")
        print(f"  Abstract: {md.get('abstract', '')[:180]}...\n")


def local_search(query: str, model: SentenceTransformer, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = TOP_K) -> None:
    """Evaluate and compare Cosine, Dot Product, and L2 distance metrics locally on pre-computed vectors."""
    query_vec = encode_query(query, model)

    # Cosine similarity calculation
    cos_scores = cosine_similarity([query_vec], embeddings)[0]
    cos_top = np.argsort(-cos_scores)[:top_k]

    # Dot product calculation (identical to Cosine since vectors are normalized)
    dot_scores = embeddings @ query_vec
    dot_top = np.argsort(-dot_scores)[:top_k]

    # L2 distance (Euclidean distance) - lower scores mean closer proximity
    l2_scores = norm(embeddings - query_vec, axis=1)
    l2_top = np.argsort(l2_scores)[:top_k]

    print("\n" + "=" * 75)
    print(" ПОРІВНЯННЯ МЕТРИК НА ЛОКАЛЬНИХ ДАНИХ")
    print("=" * 75)

    print("\n[МЕТРИКА] Cosine similarity:")
    for idx in cos_top:
        print(f"- {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")

    print("\n[МЕТРИКА] Dot product:")
    for idx in dot_top:
        print(f"- {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")

    print("\n[МЕТРИКА] L2 distance:")
    for idx in l2_top:
        print(f"- {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")

    print("\n" + "-" * 75)
    print("[АНАЛІЗ РЕЗУЛЬТАТІВ]")

    if np.array_equal(cos_top, dot_top):
        print("- Метрики Cosine та Dot Product повернули ідентичний порядок статей.")
    else:
        print("- Увага: Знайдено розбіжності між результатами Cosine та Dot Product.")

    if np.array_equal(cos_top, l2_top):
        print("- Метрика L2 відстані повернула такий самий порядок, як і Cosine схожість.")
    else:
        print("- Примітка: Порядок релевантності для метрики L2 відрізняється від Cosine.")
    print("-" * 75 + "\n")


def main() -> None:
    # Initialize connection using the environment variable safely
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ПОМИЛКА] PINECONE_API_KEY не знайдено у змінних середовища або файлі .env")
        return

    print("[СТАТУС] Ініціалізація компонентів та завантаження локальних даних...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Load local data files
    try:
        df = pd.read_parquet("data/arxiv_subset.parquet")
        embeddings = np.load("embeddings/embeddings.npy")
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося завантажити локальні файли даних: {e}")
        return

    # --- Execution of Search Scenarios ---

    # 1. Baseline semantic search
    semantic_search("teaching machines to recognize objects in pictures", index, model)

    # 2. Filtered Search A: reinforcement learning, last 5 years, category cs.LG
    semantic_search(
        "reinforcement learning",
        index,
        model,
        filters={"year": {"$gte": 2021}, "category": {"$eq": "cs.LG"}}
    )

    # 3. Filtered Search B: historical papers (before 2015)
    semantic_search(
        "machine learning",
        index,
        model,
        filters={"year": {"$lt": 2015}}
    )

    # 4. Local similarity metrics cross-examination
    local_search("teaching machines to recognize objects in pictures", model, df, embeddings)


if __name__ == "__main__":
    main()