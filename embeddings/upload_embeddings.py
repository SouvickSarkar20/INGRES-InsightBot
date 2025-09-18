# embeddings/upload_to_pinecone.py
import uuid
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from .config import PINECONE_API_KEY, INDEX_NAME, CLOUD, REGION
from .build_embeddings import build

def upload(csv_path):
    # Build embeddings
    df, locations, row_embeddings, col_embeddings, dim = build(csv_path)

    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
     pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")
    index = pc.Index(INDEX_NAME)

    # Upload in batches
    batch_size = 50
    for i in tqdm(range(0, len(row_embeddings), batch_size)):
        batch_embeds = row_embeddings[i:i+batch_size]
        batch_locs = locations[i:i+batch_size]

        vectors = [
            {
                "id": str(loc),
                "values": embed.tolist(),
                "metadata": {"location": loc}
            }
            for embed, loc in zip(batch_embeds, batch_locs)
        ]

        index.upsert(vectors=vectors)

    print("✅ Data stored in Pinecone")
