# scripts/06_hybrid_search.py
# Execution: python scripts/06_hybrid_search.py

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load environment variables from .env file
load_dotenv()

# Configuration constants
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # Number of retrieved items per stream before RRF merging
RRF_K = 60   # Constant smoothing parameter for Reciprocal Rank Fusion algorithm


def search_bm25(query: str, bm25_instance: BM25Okapi, k: int = TOP_K) -> list:
    """Perform keyword-based sparse retrieval using the BM25 algorithm."""
    tokenized_query = query.lower().split()
    scores = bm25_instance.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [{"id": f"paper_{i}", "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(top_indices)]


def search_vector(query: str, index_client: Pinecone.Index, model_instance: SentenceTransformer, k: int = TOP_K) -> list:
    """Perform dense semantic retrieval using embeddings via Pinecone vector index."""
    v = model_instance.encode(query, normalize_embeddings=True).tolist()
    try:
        res = index_client.query(vector=v, top_k=k, include_metadata=False)
        return [{"id": m['id'], "score": m['score'], "rank": r + 1} for r, m in enumerate(res['matches'])]
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося виконати векторний пошук: {e}")
        return []


def hybrid_rrf(bm25_results: list, vector_results: list, k: int = RRF_K, top_n: int = 5) -> list:
    """Merge sparse and dense retrieval streams using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}

    # Process rank lists sequentially to aggregate scores
    for rank_list in [bm25_results, vector_results]:
        for item in rank_list:
            doc_id = item['id']
            rank = item['rank']
            # Standard RRF mathematical implementation: 1 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort candidates in descending order based on their unified RRF value
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "rrf_score": score} for doc_id, score in sorted_rrf[:top_n]]


def display_hybrid_results(query: str, bm25_res: list, vec_res: list, hybrid_res: list, df_indexed: pd.DataFrame) -> None:
    """Format and display comparative search benchmarks to stdout."""
    print("\n" + "=" * 80)
    print(f" ЗАПИТ: {query}")
    print("=" * 80)

    # Render top sparse matches
    print("\n[ТОП-3 BM25 (Ключові слова)]")
    limit_bm25 = min(3, len(bm25_res))
    for i in range(limit_bm25):
        doc_id = bm25_res[i]['id']
        title = df_indexed.loc[doc_id, 'title'] if doc_id in df_indexed.index else "Unknown"
        print(f"  {i+1}. {title[:70]}...")

    # Render top dense matches
    print("\n[ТОП-3 VECTOR (Семантика)]")
    limit_vec = min(3, len(vec_res))
    for i in range(limit_vec):
        doc_id = vec_res[i]['id']
        title = df_indexed.loc[doc_id, 'title'] if doc_id in df_indexed.index else "Unknown"
        print(f"  {i+1}. {title[:70]}...")

    # Render top hybrid fusion matches
    print("\n[ГІБРИДНИЙ ТОП-5 (RRF Ранжування)]")
    for i, res in enumerate(hybrid_res):
        doc_id = res['id']
        if doc_id in df_indexed.index:
            row = df_indexed.loc[doc_id]
            print(f"  {i+1}. Score RRF: {res['rrf_score']:.4f} | {row['title']} ({row['year']})")
        else:
            print(f"  {i+1}. Score RRF: {res['rrf_score']:.4f} | [Дані не знайдено для ID: {doc_id}]")


def main() -> None:
    # 1. Authorization credentials check
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ПОМИЛКА] PINECONE_API_KEY не знайдено у змінних середовища або файлі .env")
        return

    print("[СТАТУС] Ініціалізація моделей та завантаження локальних даних...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)

    try:
        raw_df = pd.read_parquet("data/arxiv_subset.parquet")
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося завантажити файл даних: {e}")
        return

    # Create a clean data representation mapping directly to vector IDs
    df = raw_df.reset_index(drop=True)
    df.index = [f"paper_{i}" for i in range(len(df))]

    # Build the tokenized text structures for BM25 processing
    print("[СТАТУС] Побудова зворотного індексу BM25Okapi за текстовим корпусом...")
    tokenized_corpus = [
        (str(row['title']) + " " + str(row['abstract'])).lower().split()
        for _, row in df.iterrows()
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. Execution of defined verification scenarios
    queries = [
        "BERT fine-tuning",                                    # Strict technical keyword matching
        "Yann LeCun convolutional networks",                   # Proper nouns / Author query matching
        "making computers understand human emotions from text" # Pure paraphrased intent mapping
    ]

    for q in queries:
        b_res = search_bm25(q, bm25)
        v_res = search_vector(q, index, model)
        h_res = hybrid_rrf(b_res, v_res)
        display_hybrid_results(q, b_res, v_res, h_res, df)


if __name__ == "__main__":
    main()