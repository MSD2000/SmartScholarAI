# SmartScholar: ArXiv Paper Search (MiniLM)

SmartScholar is a lightweight semantic search interface for ArXiv papers. It allows users to:

1. **Search by exact paper title** (case-insensitive) and immediately retrieve that paper.
2. **Enter keywords or partial titles** to retrieve the top-K most semantically similar papers, powered by the all-MiniLM-L6-v2 SentenceTransformer model.

Each result is displayed as a “card” containing:
- **Title** (and semantic similarity score, if applicable);
- **Authors** (each author name is a hyperlink to their Google Scholar profile);
- **Abstract** (shown in full);
- **“Read via DOI”** and **“Search on Google Scholar”** hyperlinks;
- An expandable **“Show additional info”** section revealing metadata such as submitter, comments, journal reference, categories, license, update date, parsed authors, and main category;
- A help button that uses a locally deployed model (tinyllama) to explain the abstract in simple terms and give keyword examples;
- Text-to-speech (pyttsx3) for reading the help, as an accessibility option.

---

## 📁 Repository Structure

```
SmartScholarAI/
├── all-MiniLM-L6-v2/             # Optional folder if you store model locally
├── app/
│   └── smart_scholar_ai_v3.py    # Main Streamlit application
├── data/
│   ├── arxiv_tokenized_balanced.csv   # Tokenized-balanced metadata CSV
│   └── arxiv_minilm_embeddings.npy    # Precomputed MiniLM embeddings (N×384)
├── notebooks/                    # Jupyter notebooks (optional)
├── scripts/                      # Helper scripts (optional)
├── venv/                         # Virtual environment directory (optional)
├── LICENSE                       # Project license (e.g., MIT)
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

- **app/app_streamlit.py**  
  The Streamlit app that loads metadata + embeddings, handles exact‐title and semantic keyword searches, and renders results.

- **data/arxiv_tokenized_balanced.csv**  
  The tokenized‐balanced subset of ArXiv papers. Columns include:
  ```
  title, abstract, submitter, authors, comments, journal-ref, doi, report-no,
  categories, license, update_date, authors_parsed, main_category, title_norm
  ```
  (The `title_norm` column is added on load to facilitate case-insensitive title matching.)

- **data/arxiv_minilm_embeddings.npy**  
  A memory-mapped NumPy array of shape (N_papers × 384), containing all-MiniLM-L6-v2 embeddings for each row in `arxiv_tokenized_balanced.csv`.

- **requirements.txt**  
  A list of required Python packages:
  ```txt
  altair
  attrs
  blinker
  cachetools
  certifi
  charset-normalizer
  click
  colorama
  comtypes
  filelock
  fsspec
  gitdb
  GitPython
  huggingface-hub
  idna
  Jinja2
  joblib
  jsonschema
  jsonschema-specifications
  MarkupSafe
  mpmath
  narwhals
  networkx
  numpy
  packaging
  pandas
  pillow
  protobuf
  pyarrow
  pydeck
  pypiwin32
  python-dateutil
  pyttsx3
  pytz
  pywin32
  PyYAML
  referencing
  regex
  requests
  rpds-py
  safetensors
  scikit-learn
  scipy
  sentence-transformers
  six
  smmap
  streamlit
  sympy
  tenacity
  threadpoolctl
  tokenizers
  toml
  torch
  tornado
  tqdm
  transformers
  typing_extensions
  tzdata
  urllib3
  watchdog
  ```

- **LICENSE**  
  The project license (e.g., MIT) describing how the code can be used and distributed.

---

## 🛠 Dependencies

Make sure you have our prepared dataset: https://drive.google.com/file/d/15zcCRoalLWW74Eh2UwyP3ubhzQAyzU4I/view?usp=sharing
Make sure you have locally deployed Ollama with tinyllama.
Make sure you have locally downloaded inside the root project folder all-MiniLM-L6-v2 model.
Make sure you have **Python 3.10+** installed.

Install the required libraries with:

```bash
pip install -r requirements.txt
```

This will install:

- **Streamlit**: For building the interactive web interface.  
- **Pandas** & **NumPy**: For data loading and manipulation.  
- **scikit-learn**: For cosine similarity computations.  
- **sentence-transformers**: To load the all-MiniLM-L6-v2 model for embedding queries.

---

## 🚀 Running the App

1. **Clone the repository** (or download/unwrap the ZIP):

   ```bash
   git clone https://github.com/MSD2000/SmartScholarAI.git
   cd SmartScholarAI
   ```

2. **Verify data files**  
   Ensure the following files exist:

   ```
   SmartScholarAI/
   ├── app/
   │   └── smart_scholar_ai_v3.py
   └── data/
       ├── arxiv_tokenized_balanced.csv
       └── arxiv_minilm_embeddings.npy
   ```

   If you don’t have these data files, please download them from your data source and place them under `data/` exactly as named above.

3. **Activate your virtual environment** (if using one):

   ```bash
   source venv/bin/activate    # macOS/Linux
   .\venv\Scripts\activate  # Windows
   ```

4. **Install dependencies** (if you haven’t already):

   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Streamlit**:

   ```bash
   streamlit run app/smart_scholar_ai_v3.py
   ```

6. **Open your browser**  
   Streamlit will start a local server (by default at `http://localhost:8501`). Navigate there to use the search interface.

---

## 🔍 How It Works

1. **Metadata Loading**  
   - On startup, `app/app_streamlit.py` calls `pd.read_csv("data/arxiv_tokenized_balanced.csv", usecols=[…])` to load only the needed columns into a DataFrame (`df_meta`).  
   - It then adds a normalized column `title_norm = title.lower().strip()` for quick exact‐title matching.

2. **Embedding Loading (Memory-Mapped)**  
   - The precomputed NumPy embeddings are loaded using `np.load("data/arxiv_minilm_embeddings.npy", mmap_mode="r")`.  
   - Memory-mapping ensures the `.npy` file is not fully read into RAM at once.

3. **Exact‐Title Match**  
   - When the user submits a query, the app checks if `query.lower()` equals any `df_meta["title_norm"]`.  
   - If an exact match is found, those paper(s) are displayed immediately (no semantic step).

4. **Semantic Search**  
   - If no exact match, the app loads the all-MiniLM-L6-v2 model once (cached by Streamlit).  
   - The query is encoded into a 384-dim vector.  
   - Cosine similarities are computed between that query vector and each row of the memory-mapped embeddings.  
   - The top-K highest‐similarity papers are selected and displayed in descending order of similarity score.

5. **Result Rendering**  
   Each result card includes:  
   - **Title** (with a small “score” badge if semantic search)  
   - **Authors** (split on commas/“and” via a regex, then rendered inline with each author as a clickable link to their Google Scholar profile)  
   - **Full Abstract**  
   - **Links Row**:  
     - “🔗 Read via DOI” (if the paper has a DOI)  
     - “🔎 Search on Google Scholar” (searches by title)  
   - **Expandable Section** (“Show additional info”) revealing fields:  
     - `submitter`, `comments`, `journal-ref`, `report-no`, `categories`, `license`, `update_date`, `authors_parsed`, `main_category`.

---

## 🙋‍♂️ Authors

- [Vasil Kozov](https://github.com/VasilKozov) (post-doc) | vkozov@uni-ruse.bg
- [Martin Dzhurov](https://github.com/MSD2000) (masters-degree student) | mdzhurov@uni-ruse.bg
- [Kristian Spasov](https://github.com/kristianinator) (masters-degree student) | kspasov@uni-ruse.bg

University of Ruse "Angel Kanchev" - Ruse, Bulgaria
