"""Known-item retrieval evaluation (Scopus/WoS-style IR metrics) for very large corpora.

This script is an optimized rewrite of your analysis.py to avoid:
- loading a multi-GB CSV fully into RAM
- building BM25 / TF-IDF / full sorting over the full corpus

Key changes vs your previous version:
1) Works on a RANDOM SUBSET of the corpus (uniform sampling with fixed seed).
2) Loads only required CSV columns in chunks and slices embeddings via numpy memmap.
3) Computes ONLY top-k candidates (you only need k<=10 metrics).
4) Exact-match baseline uses a dictionary (no repeated full DataFrame scans).

It reports: P@1, R@1, nDCG@1, P@5, R@5, nDCG@5, P@10, R@10, nDCG@10, MRR, MAP@10
for each query type and retrieval method.

NOTE: If you later want to run on the FULL corpus, you need ANN indexing (FAISS) for dense
retrieval and an inverted index (e.g., Pyserini/Elasticsearch) for BM25.
"""

import re
import math
import time
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Optional BM25 baseline
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# Semantic embeddings (MiniLM)
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG (adjust paths here)
# -----------------------------
CSV_PATH = r"./data/arxiv_dataset.csv"
EMB_PATH = r"./data/arxiv_full_data_embeddings.npy"

# Columns to read from CSV (edit if yours differ)
TITLE_COL = "title"
ABSTRACT_COL = "abstract"
DOC_ID_COL = "doc_id"           # if not present, we will create it from global row index
MAIN_CAT_COL = "main_category"  # optional

# Subset sampling (critical for 32 GB machines with multi-GB datasets)
SUBSET_N = 200_000            # try 100_000 if still heavy; 200k is a good starting point
CHUNK_SIZE = 50_000           # CSV streaming chunk size
RNG_SEED = 42

# Evaluation settings
N_QUERIES = 300               # number of query documents sampled from the subset
TOPK_CANDIDATES = 100         # we only need top 10 metrics, but use 100 candidates for stability

# TF-IDF settings
TFIDF_MAX_FEATURES = 60_000

# BM25 settings (BM25 is expensive; keep docs shorter to save memory)
ENABLE_BM25 = True            # set False if you still run into RAM/time issues
BM25_MAX_DOC_TOKENS = 200     # cap tokens per document

# Query generation
NOISY_TITLE_DROP_PROB = 0.4
NOISY_TITLE_MIN_TERMS = 4
NOISY_TITLE_MAX_TERMS = 10
FREQ_KEYWORDS_TOP_N = 8
ABSTRACT_SNIPPET_MAX_TOKENS = 24


# -----------------------------
# Utility
# -----------------------------
STOP = set(ENGLISH_STOP_WORDS)


def simple_tokens(text: str):
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def get_embedding_row_count(emb_path: str) -> int:
    """Return number of rows in the embedding matrix without loading it into RAM."""
    emb = np.load(emb_path, mmap_mode="r")
    return int(emb.shape[0])



def make_noisy_title_query(
    title: str,
    drop_prob: float = NOISY_TITLE_DROP_PROB,
    min_terms: int = NOISY_TITLE_MIN_TERMS,
    max_terms: int = NOISY_TITLE_MAX_TERMS,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random(0)
    toks = [t for t in simple_tokens(title) if len(t) > 2]
    if not toks:
        return title or ""
    kept = [t for t in toks if rng.random() > drop_prob]
    if len(kept) < min_terms:
        kept = toks[:min_terms]
    kept = kept[:max_terms]
    return " ".join(kept)


def make_freq_keyword_query(title: str, abstract: str, top_n: int = FREQ_KEYWORDS_TOP_N) -> str:
    toks = [
        t for t in simple_tokens((title or "") + " " + (abstract or ""))
        if t not in STOP and len(t) > 2
    ]
    if not toks:
        return ""
    counts = Counter(toks)
    return " ".join([w for w, _ in counts.most_common(top_n)])


def make_abstract_snippet_query(abstract: str, max_tokens: int = ABSTRACT_SNIPPET_MAX_TOKENS) -> str:
    toks = [t for t in simple_tokens(abstract or "") if t not in STOP and len(t) > 2]
    return " ".join(toks[:max_tokens])


def topk_docids_from_scores(scores: np.ndarray, doc_ids: list[str], k: int) -> list[str]:
    """Return doc_ids for top-k scores without sorting the full array."""
    k = min(k, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [doc_ids[i] for i in idx]


# -----------------------------
# IR metrics
# -----------------------------

def precision_at_k(ranked, relevant, k):
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked, relevant):
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked, relevant, k=None):
    if not relevant:
        return 0.0
    if k is None:
        k = len(ranked)
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def dcg(rels):
    s = 0.0
    for i, rel in enumerate(rels, start=1):
        s += (2**rel - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked, rel_grade_fn, k):
    rels = [rel_grade_fn(d) for d in ranked[:k]]
    ideal_rels = sorted(rels, reverse=True)
    denom = dcg(ideal_rels)
    return 0.0 if denom == 0 else dcg(rels) / denom


# -----------------------------
# Data loading (subset)
# -----------------------------

def load_subset_csv_and_embeddings(
    csv_path: str,
    emb_path: str,
    subset_n: int,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a uniform random subset of rows from a huge CSV, and slice matching embeddings.

    Returns:
        df_subset: DataFrame for the subset, ordered by global row index
        doc_emb:   float32 embeddings aligned with df_subset
        keep_idx:  numpy array of global row indices kept (ascending)
    """

    # IMPORTANT:
    # Many arXiv-style CSV exports contain newlines inside quoted fields (especially abstracts).
    # Therefore, counting lines in the raw file is NOT a reliable way to count rows.
    # Instead, we treat the embeddings file as the "source of truth" for row count,
    # and later verify that pandas parses exactly the same number of CSV rows.
    print("Reading embedding matrix header (memmap) to get row count...")
    t0 = time.perf_counter()
    total_rows = get_embedding_row_count(emb_path)
    print(f"Embeddings rows (source of truth): {total_rows:,} (read in {time.perf_counter()-t0:.2f}s)")

    n = min(subset_n, total_rows)
    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(total_rows, size=n, replace=False))

    # Load only needed columns to minimize RAM
    # If DOC_ID_COL is missing, we will create doc_id from the global row id.
    usecols = [c for c in [DOC_ID_COL, TITLE_COL, ABSTRACT_COL, MAIN_CAT_COL] if c is not None]

    # Read header first to see what columns exist
    header = pd.read_csv(csv_path, nrows=0)
    existing_cols = set(header.columns.tolist())

    # Filter usecols to only those that exist
    usecols = [c for c in usecols if c in existing_cols]

    dtype_map = {}
    if DOC_ID_COL in usecols:
        dtype_map[DOC_ID_COL] = "string"

    print(f"Streaming CSV in chunks; keeping {n:,} rows...")
    kept_chunks = []
    start = 0

    # We need to grab specific row positions within each chunk efficiently.
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        usecols=usecols,
        low_memory=False,
        dtype=dtype_map if dtype_map else None,
    ):
        end = start + len(chunk)

        left = np.searchsorted(keep_idx, start, side="left")
        right = np.searchsorted(keep_idx, end, side="left")

        if right > left:
            rel = keep_idx[left:right] - start
            sub = chunk.iloc[rel].copy()
            sub["_global_row"] = keep_idx[left:right]
            kept_chunks.append(sub)

        start = end
        if start % (chunk_size * 10) == 0:
            kept_so_far = sum(len(x) for x in kept_chunks)
            print(f"  parsed {start:,} CSV rows so far; kept {kept_so_far:,}")

    parsed_rows = start
    if parsed_rows != total_rows:
        raise ValueError(
            f"CSV parsed rows ({parsed_rows:,}) != embeddings rows ({total_rows:,}). "
            "This usually means the CSV file does not correspond to the embedding file "
            "(different dataset/version), or the CSV has parsing issues (delimiter/quoting)."
        )

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.sort_values("_global_row").reset_index(drop=True)

    # Ensure required text columns exist
    if TITLE_COL not in df.columns:
        df[TITLE_COL] = ""
    if ABSTRACT_COL not in df.columns:
        df[ABSTRACT_COL] = ""

    df[TITLE_COL] = df[TITLE_COL].fillna("")
    df[ABSTRACT_COL] = df[ABSTRACT_COL].fillna("")

    # Ensure doc_id exists and is string
    if DOC_ID_COL not in df.columns:
        df[DOC_ID_COL] = df["_global_row"].astype("string")
    else:
        df[DOC_ID_COL] = df[DOC_ID_COL].astype("string")

    # Load embeddings as memmap and slice
    print("Loading embeddings as memmap and slicing subset...")
    emb_all = np.load(emb_path, mmap_mode="r")

    # Sanity: embeddings row count should match parsed CSV rows
    if int(emb_all.shape[0]) != parsed_rows:
        raise ValueError(
            f"Embeddings rows ({int(emb_all.shape[0]):,}) != CSV parsed rows ({parsed_rows:,}). "
            "They must align 1-to-1 in the same order."
        )

    doc_emb = np.asarray(emb_all[keep_idx], dtype=np.float32)

    # Normalize embeddings for cosine similarity via dot product
    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_emb = doc_emb / norms

    print(f"Subset df rows: {len(df):,}; subset embeddings: {doc_emb.shape}")
    return df, doc_emb, keep_idx


# -----------------------------
# Main evaluation
# -----------------------------

def main():
    rng = random.Random(RNG_SEED)

    print("\n=== CONFIG ===")
    print("CSV_PATH:", CSV_PATH)
    print("EMB_PATH:", EMB_PATH)
    print("SUBSET_N:", SUBSET_N)
    print("N_QUERIES:", N_QUERIES)
    print("TOPK_CANDIDATES:", TOPK_CANDIDATES)
    print("TFIDF_MAX_FEATURES:", TFIDF_MAX_FEATURES)
    print("rank-bm25 installed:", HAS_BM25)
    print("BM25 enabled (config):", ENABLE_BM25)

    # ---- Load subset + embeddings ----
    df, doc_emb, keep_idx = load_subset_csv_and_embeddings(
        CSV_PATH, EMB_PATH, SUBSET_N, CHUNK_SIZE, RNG_SEED
    )

    # Combined text for lexical retrieval
    df["combined_text"] = (df[TITLE_COL] + ". " + df[ABSTRACT_COL]).str.strip()

    doc_ids = df[DOC_ID_COL].tolist()
    corpus_texts = df["combined_text"].tolist()

    # ---- Build TF-IDF index ----
    print("\nBuilding TF-IDF index...")
    t0 = time.perf_counter()
    tfidf = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    X = tfidf.fit_transform(corpus_texts)
    print(f"TF-IDF built: X shape={X.shape} in {time.perf_counter()-t0:.1f}s")

    # ---- Optional BM25 index ----
    bm25 = None
    enable_bm25_runtime = HAS_BM25 and ENABLE_BM25
    if enable_bm25_runtime:
        print("\nBuilding BM25 index (tokenizing corpus)...")
        t0 = time.perf_counter()
        tokenized_corpus = []
        for text in corpus_texts:
            toks = [t for t in simple_tokens(text) if t not in STOP and len(t) > 2]
            if BM25_MAX_DOC_TOKENS is not None:
                toks = toks[:BM25_MAX_DOC_TOKENS]
            tokenized_corpus.append(toks)
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 built in {time.perf_counter()-t0:.1f}s")
    else:
        print("\nBM25 disabled at runtime (either not installed or disabled in config).")

    # ---- Semantic model ----
    print("\nLoading MiniLM model...")
    t0 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"MiniLM loaded in {time.perf_counter()-t0:.1f}s")

    # ---- Exact-match dictionary (fast baseline) ----
    print("\nBuilding exact-title lookup dictionary...")
    t0 = time.perf_counter()
    title_to_docids = defaultdict(list)
    for did, title in zip(doc_ids, df[TITLE_COL].tolist()):
        title_to_docids[(title or "").strip().lower()].append(did)
    print(f"Exact-title dict built in {time.perf_counter()-t0:.1f}s")

    # ---- Sample query documents ----
    n_queries = min(N_QUERIES, len(df))
    query_rows = rng.sample(range(len(df)), n_queries)

    queries = []
    for ridx in query_rows:
        row = df.iloc[ridx]
        q1 = make_noisy_title_query(row[TITLE_COL], rng=rng)
        q2 = make_freq_keyword_query(row[TITLE_COL], row[ABSTRACT_COL], top_n=FREQ_KEYWORDS_TOP_N)
        q3 = make_abstract_snippet_query(row[ABSTRACT_COL], max_tokens=ABSTRACT_SNIPPET_MAX_TOKENS)

        queries.append(
            {
                "qid": f"q{ridx}",
                "target_doc_id": row[DOC_ID_COL],
                "main_category": row.get(MAIN_CAT_COL, None),
                "query_noisy_title": q1,
                "query_keywords_freq": q2,
                "query_abstract_snippet": q3,
            }
        )

    # ---- Known-item relevance ----
    qrels = {q["qid"]: {q["target_doc_id"]} for q in queries}

    # ---- Optional graded relevance for nDCG ----
    if MAIN_CAT_COL in df.columns:
        doc_to_cat = dict(zip(df[DOC_ID_COL], df[MAIN_CAT_COL]))

        def make_rel_grade_fn(q):
            target = q["target_doc_id"]
            target_cat = doc_to_cat.get(target, None)

            def rel_grade(doc_id):
                if doc_id == target:
                    return 2
                if target_cat is not None and doc_to_cat.get(doc_id, None) == target_cat:
                    return 1
                return 0

            return rel_grade

    else:

        def make_rel_grade_fn(q):
            target = q["target_doc_id"]
            return lambda doc_id: 2 if doc_id == target else 0

    # ---- Methods ----
    cutoffs = [1, 5, 10]
    methods = ["exact_match", "tfidf", "minilm"]
    if enable_bm25_runtime:
        methods.insert(2, "bm25")

    # Retrieval functions (return only top candidates)
    def run_exact_match(query_text: str) -> list[str]:
        # exact title equality (lower bound)
        q = (query_text or "").strip().lower()
        hits = title_to_docids.get(q, [])
        return hits[:TOPK_CANDIDATES]

    def run_tfidf(query_text: str) -> list[str]:
        qv = tfidf.transform([query_text])
        # With default TfidfVectorizer(norm='l2'), dot product equals cosine similarity.
        scores = (X @ qv.T).toarray().ravel()
        return topk_docids_from_scores(scores, doc_ids, TOPK_CANDIDATES)

    def run_bm25(query_text: str) -> list[str]:
        toks = [t for t in simple_tokens(query_text) if t not in STOP and len(t) > 2]
        # rank-bm25 can return top-n directly (still computes scores internally)
        return bm25.get_top_n(toks, doc_ids, n=TOPK_CANDIDATES)

    def run_minilm(query_text: str) -> list[str]:
        q_emb = model.encode([query_text], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        # cosine similarity via dot product because both sides are normalized
        scores = (q_emb @ doc_emb.T).ravel()
        return topk_docids_from_scores(scores, doc_ids, TOPK_CANDIDATES)

    def eval_one_query(ranked, relevant, rel_grade_fn):
        out = {}
        for k in cutoffs:
            out[f"P@{k}"] = precision_at_k(ranked, relevant, k)
            out[f"R@{k}"] = recall_at_k(ranked, relevant, k)
            out[f"nDCG@{k}"] = ndcg_at_k(ranked, rel_grade_fn, k)
        out["MRR"] = reciprocal_rank(ranked, relevant)
        out["MAP@10"] = average_precision(ranked, relevant, k=10)
        return out

    # ---- Evaluation loop ----
    rows = []
    qtypes = ["query_noisy_title", "query_keywords_freq", "query_abstract_snippet"]

    print("\nRunning evaluation...")
    t_eval = time.perf_counter()

    for qi, q in enumerate(queries, start=1):
        for qtype in qtypes:
            qtext = q[qtype]
            relevant = qrels[q["qid"]]
            rel_grade_fn = make_rel_grade_fn(q)

            for method in methods:
                if method == "exact_match":
                    ranked = run_exact_match(qtext)
                elif method == "tfidf":
                    ranked = run_tfidf(qtext)
                elif method == "bm25":
                    ranked = run_bm25(qtext)
                elif method == "minilm":
                    ranked = run_minilm(qtext)
                else:
                    continue

                m = eval_one_query(ranked, relevant, rel_grade_fn)
                rows.append({"method": method, "query_type": qtype, **m})

        if qi % 25 == 0:
            elapsed = time.perf_counter() - t_eval
            print(f"  progress: {qi}/{len(queries)} queries done ({elapsed:.1f}s elapsed)")

    res = pd.DataFrame(rows)
    summary = res.groupby(["query_type", "method"]).mean(numeric_only=True).reset_index()

    print("\n=== Aggregated results (mean over queries) ===")
    print(summary.to_string(index=False))

    # Save for easy pasting into the paper
    out_csv = "./evaluation_results_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()
