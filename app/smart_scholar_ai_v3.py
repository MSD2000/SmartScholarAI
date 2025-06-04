import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import pyttsx3

@st.cache_data
def load_tokenized_meta(path_csv: str) -> pd.DataFrame:
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
    ]
    df = pd.read_csv(path_csv, usecols=use_cols, low_memory=False)
    df["title_norm"] = df["title"].str.strip().str.lower()
    return df

@st.cache_data
def load_minilm_embeddings(path_npy: str) -> np.ndarray:
    return np.load(path_npy, mmap_mode="r")

@st.cache_data(show_spinner=False)
def get_minilm_model():
    return SentenceTransformer(os.path.abspath(os.path.join(os.path.dirname(__file__), "../all-MiniLM-L6-v2")))

def split_authors(authors_str: str) -> list[str]:
    if not isinstance(authors_str, str) or not authors_str.strip():
        return []
    pieces = re.split(r'\s+and\s+|,\s*', authors_str)
    return [p.strip() for p in pieces if p.strip()]

def main():
    if "tts_active" not in st.session_state:
        st.session_state.tts_active = False
    st.set_page_config(page_title="SmartScholar: MiniLM Search", layout="centered")
    st.title("SmartScholar: ArXiv Paper Search (RawData + MiniLM)")

    tokenized_csv = "../data/arxiv_dataset.csv"
    embeddings_npy = "../data/arxiv_full_data_embeddings.npy"

    df_meta = load_tokenized_meta(tokenized_csv)
    embeddings = load_minilm_embeddings(embeddings_npy)

    if df_meta.shape[0] != embeddings.shape[0]:
        st.error(
            f"⚠ Metadata rows ({df_meta.shape[0]}) do not match embeddings rows ({embeddings.shape[0]})!\n"
            "Ensure files are synced and aligned."
        )
        st.stop()

    # Inputs
    query = st.text_input("Enter a paper title or keywords:")
    top_k = st.slider("Number of results to display:", 1, 20, 5)

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.query = ""

    if st.button("Search"):
        q = query.strip()
        if not q:
            st.warning("Please enter a search query.")
            return

        st.session_state.query = q
        q_norm = q.lower()
        exact_matches = df_meta[df_meta["title_norm"] == q_norm]

        if not exact_matches.empty:
            st.session_state.results = exact_matches
        else:
            st.info("Running semantic search...")
            model = get_minilm_model()
            q_embedding = model.encode([q], show_progress_bar=False)
            sims = cosine_similarity(q_embedding, embeddings).flatten()
            top_indices = np.argsort(sims)[-top_k:][::-1]
            df_subset = df_meta.iloc[top_indices].copy()
            df_subset["score"] = sims[top_indices]
            st.session_state.results = df_subset

    # Display stored results if present
    if st.session_state.results is not None:
        if "score" in st.session_state.results.columns:
            st.subheader(f"Top {top_k} Semantic Results for: \"{st.session_state.query}\"")
            for i, (_, row) in enumerate(st.session_state.results.iterrows()):
                render_result(row, show_score=True, idx=i)
        else:
            st.subheader("Exact Title Match")
            for i, (_, row) in enumerate(st.session_state.results.iterrows()):
                render_result(row, show_score=False, idx=i)


def render_result(row: pd.Series, show_score: bool, idx: int):
    with st.container():
        if show_score:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {row['title']}")
            with col2:
                st.markdown(f"<span style='color:gray; font-size:14px;'>score: {row['score']:.3f}</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown(f"### {row['title']}")

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

        link_col1, link_col2 = st.columns([1, 1])
        with link_col1:
            gs_query = urllib.parse.quote_plus(row["title"])
            gs_url = f"https://scholar.google.com/scholar?q={gs_query}"
            st.markdown(f":mag: [Search on Google Scholar]({gs_url})")
        with link_col2:
            doi = row.get("doi", "")
            if pd.notna(doi) and doi.strip():
                st.markdown(f":link: [Read via DOI](https://doi.org/{doi})")
        # Create a unique key per result row
        llm_key = f"llm_response_{idx}"
        explain_btn_key = f"explain_button_{idx}"
        speak_btn_key = f"speak_button_{idx}"

        # First button: trigger explanation
        if st.button("💡 Help me understand this better", key=explain_btn_key):
            with st.spinner("Querying LLM..."):
                author = row.get("authors", "Unknown")
                abstract = row.get("abstract", "")
                response = query_llm(author, abstract)
                st.session_state[llm_key] = response

        # If we already have a response, show it and enable speech
        if llm_key in st.session_state:
            st.markdown("### Explanation")
            st.info(st.session_state[llm_key])

            if st.button("🔊 Read aloud", key=speak_btn_key):
                speak_text(st.session_state[llm_key])

        st.markdown("**Abstract:**")
        st.markdown(row["abstract"])

        with st.expander("Show additional info"):
            st.write(f"**Submitter:** {row.get('submitter', 'N/A')}")
            st.write(f"**Comments:** {row.get('comments', 'N/A')}")
            st.write(f"**Journal-Ref:** {row.get('journal-ref', 'N/A')}")
            st.write(f"**Report-No:** {row.get('report-no', 'N/A')}")
            st.write(f"**Categories:** {row.get('categories', 'N/A')}")
            st.write(f"**License:** {row.get('license', 'N/A')}")
            st.write(f"**Update Date:** {row.get('update_date', 'N/A')}")
            st.write(f"**Authors Parsed:** {row.get('authors_parsed', 'N/A')}")

        st.markdown("---")

def query_llm(author, abstract):
    if not author or not abstract:
        return "Missing author or abstract"

    prompt = f"""
Explain this abstract in simple terms

Abstract:
"{abstract}"
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": prompt, "stream": False},
            timeout=60
        )
        return response.json().get("response", "No response from LLM.")
    except Exception as e:
        return f"Error: {e}"

def speak_text(text):
    try:
        engine = pyttsx3.init()
        st.session_state.tts_active = True
        engine.say(text)
        if st.session_state.tts_active:
            engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

if __name__ == "__main__":
    main()
