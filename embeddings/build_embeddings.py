# embeddings/build_embeddings.py
from .utils import load_data, get_embedder, create_embeddings

def build(csv_path):
    df, locations, info_rows, columns = load_data(csv_path)
    embed_model = get_embedder()
    row_embeddings, col_embeddings = create_embeddings(embed_model, info_rows, columns)
    dim = row_embeddings.shape[1]
    return df, locations, row_embeddings, col_embeddings, dim

