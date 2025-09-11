import os
import uuid
import pandas as pd
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# -------------------------------
# 1. Load ENV + CSV
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "chatbot-index"
CLOUD = "aws"
REGION = "us-east-1"

csv_path = "../data/cleaned_data.csv"
df = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df)} records")

locations = df.iloc[:, 0].tolist()  # first col = location (ID)
info_rows = df.iloc[:, 1:].astype(str).agg(" ".join, axis=1).tolist()
columns = df.columns[1:].tolist()

# Use lightweight embedding model (CPU-friendly)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  
embeddings = embed_model.encode(info_rows, convert_to_numpy=True, show_progress_bar=True)
col_embeddings = embed_model.encode(columns,convert_to_numpy=True)

DIMENSION = embeddings.shape[1]
print(f"✅ Embeddings created with dimension {DIMENSION}")

pc = Pinecone(api_key=PINECONE_API_KEY)

if "chatbot-index" in pc.list_indexes().names():
    pc.delete_index("chatbot-index")
    print("Old index deleted.")

print("old index deleted")

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# -------------------------------
# 4. Upsert into Pinecone
# -------------------------------
batch_size = 50
for i in tqdm(range(0, len(embeddings), batch_size)):
    batch_embeds = embeddings[i:i+batch_size]
    batch_locs = locations[i:i+batch_size]

    vectors = []
    for embed, loc in zip(batch_embeds, batch_locs):
        vectors.append({
            "id": str(loc),
            "values": embed.tolist(),
            "metadata": {"location": loc}
        })

    index.upsert(vectors=vectors)

print("✅ Data stored in Pinecone")