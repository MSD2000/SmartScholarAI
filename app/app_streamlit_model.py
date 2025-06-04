import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# CACHED HELPERS
# -----------------------------------------
@st.cache_data
def load_tokenized_df(path: str) -> pd.DataFrame:
    """
    Load the tokenized dataset CSV into a pandas DataFrame.
    Expects at least these columns:
      - 'id'
      - 'combined_text'
      - 'title'
      - 'abstract'
      - 'authors'
      - 'doi'
    """
    # low_memory=False silences mixed‐type warnings
    df = pd.read_csv(path, low_memory=False)
    return df

@st.cache_data
def build_tfidf_matrix(texts: pd.Series) -> tuple:
    """
    Build and cache a TF-IDF vectorizer + matrix given a series of texts.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

@st.cache_data
def load_numpy_embeddings(npy_path: str) -> np.ndarray:
    """
    Load and cache a .npy matrix of shape (N_papers, dim).
    """
    return np.load(npy_path)

@st.cache_data
def load_metadata_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and cache the metadata CSV for embeddings.
    Must have at least columns: ['id', 'title', 'abstract', 'authors', 'doi']
    and the same number of rows (in the same order) as the .npy embedding file.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return df

@st.cache_data(show_spinner=False)
def get_sentence_transformer_model(model_name: str):
    """
    Load and cache a SentenceTransformer model so that we can encode the query.
    Examples of model_name:
      - 'all-MiniLM-L6-v2'
      - 'allenai/specter'
      - 'allenai/scibert_scivocab_uncased'
    """
    return SentenceTransformer(model_name)

# -----------------------------------------
# MAIN APP
# -----------------------------------------
def main():
    st.set_page_config(page_title="SmartScholar Semantic Search", layout="wide")
    st.title("SmartScholar: Semantic Search / Recommender Demo")

    # --- Sidebar: let user pick method (default = MiniLM) ---
    st.sidebar.header("Choose retrieval method")
    method = st.sidebar.selectbox(
        "Method",
        (
            "🔬 MiniLM (all-MiniLM-L6-v2)",
            "🔍 TF-IDF (baseline)",
            "📄 SPECTER (allenai/specter)",
            "🧪 SciBERT (allenai/scibert_scivocab_uncased)",
        ),
        index=0,  # MiniLM is selected by default
    )

    # --- Common load: tokenized CSV ---
    tokenized_csv_path = "../data/arxiv_tokenized_balanced.csv"
    df_tokenized = load_tokenized_df(tokenized_csv_path)

    # Ensure required columns exist
    required_cols = ["id", "combined_text", "title", "abstract", "authors", "doi"]
    for col in required_cols:
        if col not in df_tokenized.columns:
            st.error(f"Missing column '{col}' in {tokenized_csv_path}")
            st.stop()

    # If method == TF-IDF, build TF-IDF matrix now:
    if method.startswith("🔍"):
        combined_texts = df_tokenized["combined_text"].fillna("")
        tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(combined_texts)

    # If method is an embedding, load the appropriate .npy and CSV
    elif method.startswith("🔬"):  # MiniLM
        embeddings = load_numpy_embeddings("../data/arxiv_minilm_embeddings.npy")
        metadata = load_metadata_csv("../data/arxiv_minilm_embeddings.csv")
    elif method.startswith("📄"):  # SPECTER
        embeddings = load_numpy_embeddings("../data/arxiv_specter_embeddings.npy")
        metadata = load_metadata_csv("../data/arxiv_specter_embeddings.csv")
    elif method.startswith("🧪"):  # SciBERT
        embeddings = load_numpy_embeddings("../data/arxiv_scibert_embeddings.npy")
        metadata = load_metadata_csv("../data/arxiv_scibert_embeddings.csv")
    else:
        st.error("Unknown method selected.")
        st.stop()

    # --- Input area on main page ---
    query = st.text_input("Enter keywords or a paper title to search:")
    top_k = st.slider("Number of results to display:", min_value=1, max_value=20, value=5, step=1)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please type at least one keyword or partial paper title.")
            st.stop()

        if method.startswith("🔍"):  # TF-IDF retrieval
            query_vec = tfidf_vectorizer.transform([query])
            cosine_sim = linear_kernel(query_vec, tfidf_matrix).flatten()
            top_indices = cosine_sim.argsort()[-top_k:][::-1]
            results_df = df_tokenized.iloc[top_indices].copy()
            results_df["score"] = cosine_sim[top_indices]

        else:
            model_name = {
                "🔬 MiniLM (all-MiniLM-L6-v2)": "all-MiniLM-L6-v2",
                "📄 SPECTER (allenai/specter)": "allenai/specter",
                "🧪 SciBERT (allenai/scibert_scivocab_uncased)": "allenai/scibert_scivocab_uncased",
            }[method]
            st.info(f"Encoding query with `{model_name}` ...")
            sbert_model = get_sentence_transformer_model(model_name)
            query_embedding = sbert_model.encode([query], show_progress_bar=False)
            cosine_sim = cosine_similarity(query_embedding, embeddings).flatten()
            top_indices = cosine_sim.argsort()[-top_k:][::-1]
            results_meta = metadata.iloc[top_indices].copy()
            results_meta["score"] = cosine_sim[top_indices]

            missing_cols = [c for c in ["abstract", "authors", "doi"] if c not in results_meta.columns]
            if missing_cols:
                results_df = results_meta.merge(
                    df_tokenized[["id", "abstract", "authors", "doi"]],
                    on="id",
                    how="left"
                )
            else:
                results_df = results_meta

        # --- Display the results ---
        st.success(f"Found top {top_k} papers using ‘{method}’")
        for _, row in results_df.iterrows():
            st.markdown(f"#### {row['title']}")
            st.markdown(f"*Score: {row['score']:.4f}*")

            # DOI link
            doi = row.get("doi", "")
            if pd.notna(doi) and doi.strip():
                st.markdown(f"[Read Full Paper](https://doi.org/{doi})")
            else:
                if "url" in row and pd.notna(row["url"]):
                    st.markdown(f"[View Full Paper]({row['url']})")

            # Google Scholar search link for the full paper title
            title_for_search = row["title"]
            if isinstance(title_for_search, str) and title_for_search.strip():
                gs_query = urllib.parse.quote_plus(title_for_search)
                gs_url = f"https://scholar.google.com/scholar?q={gs_query}"
                st.markdown(f"[Search on Google Scholar]({gs_url})")

            st.markdown(f"**Authors:** {row.get('authors','Unknown')}")
            st.markdown(f"{row.get('abstract','No abstract available')}")
            st.markdown("---")

    # Small footer
    st.markdown(
        "<sub>Built with Streamlit & SentenceTransformers</sub>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
