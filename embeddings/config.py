# embeddings/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "chatbot-index"
CLOUD = "aws"
REGION = "us-east-1"

CSV_PATH_state = "../data/cleaned_data.csv"
CSV_PATH_district = "../data/cleaned_district_level_data.csv"
