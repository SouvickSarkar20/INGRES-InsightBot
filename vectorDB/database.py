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

from langchain.chat_models import ChatGrog
from langchain.schema import HumanMessage

GROQ_API_KEY = os.getenv("GROG_API_KEY")

# ✅ Initialize LLM with Groq API
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # free Groq model
    temperature=0,
    api_key=GROQ_API_KEY
)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n💬 Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("\nUser:")
    if user_input.lower()  in ["exit","quit"]:
        break

    prompt = f"""
    The user wants to query a dataset
    Identify the location (first column) and the feature/column name for this query:
    "{user_input}"
    Provide output as JOSN : {{"location" : "<location>" , "column" : "<column>"}}
    """
    response = llm([HumanMessage(content=prompt)])

    try:
        parsed = eval(response.content)
        loc_query = parsed.get("location","")
        col_query = parsed.get("column","")
    except:    
        print("LLM failed to parse the query")
        continue

    #semantic search for location
    loc_embedding = embed_model.encode([loc_query],convert_to_numpy = True)
    loc_results = index.query(vector=loc_embedding.to_list() , top_k = 1, include_metadata=True)
    if loc_results['matches']:
        matched_loc = loc_results["matches"][0]['metadata']['locatiion']
    else:
        print("Location not found")
        continue     

    col_embedding = embed_model.encode([col_query], convert_to_numpy=True)[0]
    sims = [cosine_sim(col_embedding, c_emb) for c_emb in col_embeddings]
    best_col_idx = np.argmax(sims)
    matched_col = columns[best_col_idx]

    # 4️⃣ Fetch value from dataframe
    row = df.loc[df.iloc[:, 0] == matched_loc]
    if not row.empty:
        value = row[matched_col].values[0]
        print(f"📊 {matched_col} for {matched_loc}: {value}")
    else:
        print("Data not found for this location.")

