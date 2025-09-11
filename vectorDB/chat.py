import os
import re 
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# -------------------------------
# 1. Load ENV + Setup
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "chatbot-index"
csv_path = "../data/cleaned_data.csv"
df = pd.read_csv(csv_path)

locations = df.iloc[:, 0].tolist()
columns = df.columns[1:].tolist()

# ✅ Embedding model for queries
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
col_embeddings = embed_model.encode(columns, convert_to_numpy=True)
location_embeddings = embed_model.encode(locations, convert_to_numpy=True)

# ✅ Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    temperature=0,
    api_key=GEMINI_API_KEY
)

def cosine_sim(a, b, eps=1e-10):
    return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps)

def best_match(query_embedding, embeddings, items):
    sims = [cosine_sim(query_embedding, emb) for emb in embeddings]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    return items[best_idx], best_score

print("\n💬 Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"""
    You are a JSON generator.
    Extract from the user query:
    - location (first column name in dataset)
    - column (one of the dataset features)

    User query: "{user_input}"

    Return ONLY valid JSON, with no explanations, no extra text:
    {{
      "location": "<location>",
      "column": "<column>"
    }}
    """

    response = llm.invoke(prompt)

    try:
        raw = response.content.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        loc_query = parsed.get("location", "")
        col_query = parsed.get("column", "")
        print("✅ Parsed:", parsed)
    except Exception as e:
        print("⚠️ LLM failed to parse the query:", e)
        print("Raw response:", response.content)
        continue

    # 1️⃣ Location matching
    matched_loc = None
    if loc_query in locations:
        matched_loc = loc_query
        loc_score = 1.0
    else:
        loc_embedding = embed_model.encode([loc_query], convert_to_numpy=True)[0]
        matched_loc, loc_score = best_match(loc_embedding, location_embeddings, locations)

    if matched_loc is None or loc_score < 0.70:  # stricter threshold
        print(f"⚠️ Could not confidently identify location for '{loc_query}'. Best guess: {matched_loc} (score={loc_score:.2f})")
        continue

    # 2️⃣ Column matching with adaptive threshold
    col_embedding = embed_model.encode([col_query], convert_to_numpy=True)[0]
    matched_col, col_score = best_match(col_embedding, col_embeddings, columns)

    if matched_col is None:
        print("⚠️ Could not identify a valid column/feature from your query.")
        continue

    if col_score < 0.35:  # very weak
        print(f"⚠️ Could not confidently identify column for '{col_query}'. Best guess: {matched_col} (score={col_score:.2f})")
        continue
    elif col_score < 0.55:  # borderline → ask user
        confirm = input(f"❓ Did you mean '{matched_col}' for '{col_query}'? (y/n): ").strip().lower()
        if confirm != "y":
            print("❌ Column not confirmed. Try rephrasing your query.")
            continue

    # 3️⃣ Fetch value from dataframe
    row = df.loc[df.iloc[:, 0] == matched_loc]
    if not row.empty:
        value = row[matched_col].values[0]
        print(f"📊 {matched_col} for {matched_loc}: {value}")
    else:
        print("⚠️ Data not found for this location.")
