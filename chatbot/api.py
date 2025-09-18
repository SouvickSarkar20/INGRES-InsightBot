# chatbot/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.main import get_chatbot_response  # <-- import your unified chatbot function

app = FastAPI(title="INGRES Chatbot API", version="1.0.0")

# Request model (what frontend sends)
class ChatRequest(BaseModel):
    query: str

# Response model (what API returns)
class ChatResponse(BaseModel):
    response: str

# Health check endpoint (optional, for testing)
@app.get("/")
def root():
    return {"message": "Chatbot API is running 🚀"}

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Call chatbot core function
    result = get_chatbot_response(request.query)
    return {"response": result}
