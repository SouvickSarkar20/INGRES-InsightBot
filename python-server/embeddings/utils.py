# embeddings/utils.py
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_data(csv_path: str):
    """Load CSV and prepare locations + info rows + column names."""
    df = pd.read_csv(csv_path)
    locations = df.iloc[:, 0].tolist()  # first column = location
    info_rows = df.iloc[:, 1:].astype(str).agg(" ".join, axis=1).tolist()
    columns = df.columns[1:].tolist()
    return df, locations, info_rows, columns

def get_embedder(model_name="all-MiniLM-L6-v2"):
    """Load SentenceTransformer model."""
    return SentenceTransformer(model_name)

def create_embeddings(embed_model, info_rows, columns):
    """Generate embeddings for rows and columns."""
    row_embeddings = embed_model.encode(info_rows, convert_to_numpy=True, show_progress_bar=True)
    col_embeddings = embed_model.encode(columns, convert_to_numpy=True)
    return row_embeddings, col_embeddings
