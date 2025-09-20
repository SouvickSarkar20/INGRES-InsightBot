# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.main import get_chatbot_response  # your chatbot core logic

# ---------------------------
# 🚀 FastAPI App Initialization
# ---------------------------
app = FastAPI(
    title="INGRES InsightBot API",
    description="API for chatbot handling state/district queries using embeddings + LLM",
    version="1.0.0"
)

# ---------------------------
# 📌 Pydantic Models (Request/Response)
# ---------------------------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# ---------------------------
# 🟢 Health Check Endpoint
# ---------------------------
@app.get("/")
def root():
    return {"message": "Chatbot API is running 🚀"}

# ---------------------------
# 💬 Chat Endpoint
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = get_chatbot_response(request.query)
    return {"response": result}
