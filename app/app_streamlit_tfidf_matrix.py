import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Cache the data loading function
@st.cache_data
def load_tokenized_data(path: str) -> pd.DataFrame:
    """
    Load the tokenized dataset CSV into a pandas DataFrame.
    Expects a column 'combined_text'.
    """
    df = pd.read_csv(path)
    return df

# Cache the TF-IDF matrix building function
@st.cache_data
def build_tfidf_matrix(texts: pd.Series) -> tuple:
    """
    Build a TF-IDF vectorizer and matrix from a pandas Series of text.
    Returns (vectorizer, tfidf_matrix).
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def main():
    st.title("SmartScholar Tokenized Search")

    # Path to the tokenized data
    data_path = "../data/arxiv_tokenized_balanced.csv"
    df = load_tokenized_data(data_path)

    # Check for 'combined_text' column
    if "combined_text" not in df.columns:
        st.error("Column 'combined_text' not found in the dataset.")
        return

    # Build the TF-IDF matrix
    combined_texts = df["combined_text"].fillna("")
    vectorizer, tfidf_matrix = build_tfidf_matrix(combined_texts)

    # User inputs
    query = st.text_input("Enter search terms or keywords")
    top_k = st.slider("Number of results to display", 1, 20, 5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a valid search query.")
        else:
            # Transform query and compute cosine similarities
            query_vec = vectorizer.transform([query])
            cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()

            # Get top-k indices
            related_indices = cosine_similarities.argsort()[-top_k:][::-1]
            results = df.iloc[related_indices]

            # Display results
            for _, row in results.iterrows():
                st.subheader(row.get("title", "No Title Available"))
                st.write(row.get("abstract", "No abstract available"))
                st.write(f"**Authors:** {row.get('authors', 'Unknown')}")
                st.markdown("---")

if __name__ == "__main__":
    main()
