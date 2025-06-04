import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# CACHED HELPERS
# -----------------------------------------
@st.cache_data
def load_tokenized_meta(path_csv: str) -> pd.DataFrame:
    """
    Load only the columns we need from arxiv_tokenized_balanced.csv.
    We assume this CSV was the exact input to generate arxiv_minilm_embeddings.npy,
    so the row ordering matches the .npy exactly.

    We load:
      - title           (object)
      - abstract        (object)
      - submitter       (object)
      - authors         (object)
      - comments        (object)
      - journal-ref     (object)
      - doi             (object)
      - report-no       (object)
      - categories      (object)
      - license         (object)
      - update_date     (object)
      - authors_parsed  (object)
      - main_category   (object)
    """
    use_cols = [
        "title",
        "abstract",
        "submitter",
        "authors",
        "comments",
        "journal-ref",
        "doi",
        "report-no",
        "categories",
        "license",
        "update_date",
        "authors_parsed",
        #"main_category",
    ]
    df = pd.read_csv(path_csv, usecols=use_cols, low_memory=False)
    # Normalize title for exact-match lookups
    df["title_norm"] = df["title"].str.strip().str.lower()
    return df

@st.cache_data
def load_minilm_embeddings(path_npy: str) -> np.ndarray:
    """
    Memory‐map the MiniLM .npy file so it never loads fully into RAM.
    We assume shape = (N_papers, 384) and that N_papers == number of rows in tokenized CSV.
    """
    return np.load(path_npy, mmap_mode="r")

@st.cache_data(show_spinner=False)
def get_minilm_model():
    """Load all-MiniLM-L6-v2 only when needed."""
    return SentenceTransformer(os.path.abspath(os.path.join(os.path.dirname(__file__), "../all-MiniLM-L6-v2")))


# -----------------------------------------
# UTILITY: Robust author splitting
# -----------------------------------------
def split_authors(authors_str: str) -> list[str]:
    """
    Split an author string on commas or the word 'and' (with spaces around).
    Handles cases like:
      - "Wael Abu-Shammala and Alberto Torchinsky"
      - "Alice Smith, Bob Johnson, and Carol Lee"
      - "Alice Smith, Bob Johnson, Carol Lee"
    Returns a list of cleaned author names.
    """
    if not isinstance(authors_str, str) or not authors_str.strip():
        return []
    pieces = re.split(r'\s+and\s+|,\s*', authors_str)
    return [p.strip() for p in pieces if p.strip()]


# -----------------------------------------
# MAIN APP
# -----------------------------------------
def main():
    st.set_page_config(page_title="SmartScholar: MiniLM Search", layout="centered")
    st.title("SmartScholar: ArXiv Paper Search (RawData + MiniLM)")

    # — Adjust these paths if your folder structure differs —
    tokenized_csv  = "../data/arxiv_dataset.csv"
    embeddings_npy = "../data/arxiv_full_data_embeddings.npy"
    # —————————————————————————

    # 1) Load tokenized metadata from the “balanced” CSV
    df_meta = load_tokenized_meta(tokenized_csv)

    # 2) Memory-map the MiniLM embeddings (.npy)
    embeddings = load_minilm_embeddings(embeddings_npy)

    # Sanity check: row-count must match
    if df_meta.shape[0] != embeddings.shape[0]:
        st.error(
            f"⚠ Metadata rows ({df_meta.shape[0]}) do not match embeddings rows ({embeddings.shape[0]})!\n"
            "Ensure that `arxiv_tokenized_balanced.csv` and `arxiv_minilm_embeddings.npy` "
            "were generated from the same data in the same order."
        )
        st.stop()

    # — User Input —
    query = st.text_input("Enter a paper title or keywords:")
    top_k = st.slider("Number of results to display:", 1, 20, 5)

    if st.button("Search"):
        q = query.strip()
        if not q:
            st.warning("Please enter a paper title or some keywords.")
            st.stop()

        # 1) Exact-title match (case-insensitive)
        q_norm = q.lower()
        exact_matches = df_meta[df_meta["title_norm"] == q_norm]

        if not exact_matches.empty:
            st.subheader("Exact Title Match")
            for idx, row in exact_matches.iterrows():
                render_result(row, show_score=False)
            return  # Skip semantic ranking

        # 2) Semantic search using MiniLM embeddings
        st.subheader(f"Top {top_k} Results (Semantic Search)")
        st.info("Encoding query with all-MiniLM-L6-v2…")
        minilm_model = get_minilm_model()
        q_embedding = minilm_model.encode([q], show_progress_bar=False)

        sims = cosine_similarity(q_embedding, embeddings).flatten()
        top_indices = np.argsort(sims)[-top_k:][::-1]

        for rank, idx in enumerate(top_indices, start=1):
            row = df_meta.iloc[idx]
            row["score"] = sims[idx]
            render_result(row, show_score=True)


def render_result(row: pd.Series, show_score: bool):
    """
    Renders a single search result in a compact, card-like layout.
    If show_score=True, displays the semantic score in the top-right.
    """
    # Container for each result
    with st.container():
        # Header: Title + optional score
        if show_score:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {row['title']}")
            with col2:
                st.markdown(f"<span style='color:gray; font-size:14px;'>score: {row['score']:.3f}</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown(f"### {row['title']}")

        # Authors: inline list of clickable GS profile links
        authors = split_authors(row["authors"])
        if authors:
            author_links = []
            for a in authors:
                gs_author = urllib.parse.quote_plus(a)
                gs_auth_url = (
                    "https://scholar.google.com/"
                    f"citations?view_op=search_authors&mauthors={gs_author}"
                )
                author_links.append(f"[{a}]({gs_auth_url})")
            st.markdown("**Authors:** " + " · ".join(author_links))

        # Abstract (full)
        st.markdown(f"**Abstract:**  ")
        st.markdown(row["abstract"])

        # Links: DOI + Scholar (side by side)
        link_col1, link_col2 = st.columns([1, 1])
        with link_col1:
            gs_query = urllib.parse.quote_plus(row["title"])
            gs_url = f"https://scholar.google.com/scholar?q={gs_query}"
            st.markdown(f"🔎 [Search on Google Scholar]({gs_url})")
        with link_col2:
            doi = row.get("doi", "")
            if pd.notna(doi) and doi.strip():
                st.markdown(f"🔗 [Read via DOI](https://doi.org/{doi})")

        # Expandable “Additional Info” section
        with st.expander("Show additional info"):
            st.write(f"**Submitter:** {row.get('submitter', 'N/A')}")
            st.write(f"**Comments:** {row.get('comments', 'N/A')}")
            st.write(f"**Journal-Ref:** {row.get('journal-ref', 'N/A')}")
            st.write(f"**Report-No:** {row.get('report-no', 'N/A')}")
            st.write(f"**Categories:** {row.get('categories', 'N/A')}")
            st.write(f"**License:** {row.get('license', 'N/A')}")
            st.write(f"**Update Date:** {row.get('update_date', 'N/A')}")
            st.write(f"**Authors Parsed:** {row.get('authors_parsed', 'N/A')}")
            #st.write(f"**Main Category:** {row.get('main_category', 'N/A')}")

        st.markdown("---")


if __name__ == "__main__":
    main()
