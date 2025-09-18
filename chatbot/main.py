# chatbot/api.py

import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------------
# 1️⃣ Environment & Setup
# -------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "chatbot-index"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH_district = os.path.join(BASE_DIR, "data", "cleaned_district_level_data.csv")
CSV_PATH_state = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

# -------------------------------
# 2️⃣ Helpers
# -------------------------------
def load_and_embed(csv_path):
    df = pd.read_csv(csv_path)
    locations = df.iloc[:, 0].tolist()
    columns = df.columns[1:].tolist()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    col_embeddings = embed_model.encode(columns, convert_to_numpy=True)
    location_embeddings = embed_model.encode(locations, convert_to_numpy=True)
    return df, locations, columns, embed_model, location_embeddings, col_embeddings

def cosine_sim(a, b, eps=1e-10):
    return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps)

def best_match(query_embedding, embeddings, items):
    sims = [cosine_sim(query_embedding, emb) for emb in embeddings]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    return items[best_idx], best_score

# -------------------------------
# 3️⃣ Load Data + Models
# -------------------------------
df_district, locs_district, cols_district, embed_model_d, loc_emb_d, col_emb_d = load_and_embed(CSV_PATH_district)
df_state, locs_state, cols_state, embed_model_s, loc_emb_s, col_emb_s = load_and_embed(CSV_PATH_state)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=GEMINI_API_KEY
)

# -------------------------------
# 4️⃣ Core Chatbot Function
# -------------------------------
def chatbot_response(user_input: str) -> str:
    """Process a user query and return chatbot's response."""

    # 1️⃣ Determine scope
    classification_prompt = f"""
    Determine if the user's query refers to a 'district' or a 'state'.
    Return ONLY the word 'district' or 'state'.
    
    Query: "{user_input}"
    """
    scope = llm.invoke(classification_prompt).content.strip().lower()
    if scope == "state":
        df, locations, columns, embed_model, loc_emb, col_emb = df_state, locs_state, cols_state, embed_model_s, loc_emb_s, col_emb_s
    else:
        df, locations, columns, embed_model, loc_emb, col_emb = df_district, locs_district, cols_district, embed_model_d, loc_emb_d, col_emb_d

    # 2️⃣ Extract location & column
    prompt = f"""
    You are a JSON generator.
    Extract from the user query:
    - location (first column name in dataset)
    - column (one of the dataset features)

    User query: "{user_input}"

    Return ONLY valid JSON:
    {{
      "location": "<location>",
      "column": "<column>"
    }}
    """
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        loc_query = parsed.get("location", "")
        col_query = parsed.get("column", "")
    except Exception:
        return "⚠️ Could not parse your query. Please rephrase."

    # 3️⃣ Location matching
    if loc_query in locations:
        matched_loc, loc_score = loc_query, 1.0
    else:
        loc_embedding = embed_model.encode([loc_query], convert_to_numpy=True)[0]
        matched_loc, loc_score = best_match(loc_embedding, loc_emb, locations)

    if matched_loc is None or loc_score < 0.35:
        return f"⚠️ Could not confidently identify location for '{loc_query}'."
    elif loc_score < 0.70:
        return f"❓ Did you mean '{matched_loc}' for '{loc_query}'? (score={loc_score:.2f})"

    # 4️⃣ Column matching
    col_embedding = embed_model.encode([col_query], convert_to_numpy=True)[0]
    matched_col, col_score = best_match(col_embedding, col_emb, columns)

    if matched_col is None or col_score < 0.35:
        return f"⚠️ Could not confidently identify column for '{col_query}'."
    elif col_score < 0.70:
        return f"❓ Did you mean '{matched_col}' for '{col_query}'? (score={col_score:.2f})"

    # 5️⃣ Fetch result
    row = df.loc[df.iloc[:, 0] == matched_loc]
    if not row.empty:
        value = row[matched_col].values[0]
        return f"📊 {matched_col} for {matched_loc}: {value}"
    else:
        return "⚠️ Data not found for this location."

# -------------------------------
# 5️⃣ FastAPI Server
# -------------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(req: QueryRequest):
    response = chatbot_response(req.query)
    return {"response": response}
