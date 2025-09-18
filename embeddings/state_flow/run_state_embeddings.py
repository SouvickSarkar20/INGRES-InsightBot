from embeddings.config import CSV_PATH_state as STATE_CSV_State
from embeddings.upload_embeddings import upload

def run_state():
    upload(csv_path=STATE_CSV_State)

if __name__ == "__main__":
    run_state()
