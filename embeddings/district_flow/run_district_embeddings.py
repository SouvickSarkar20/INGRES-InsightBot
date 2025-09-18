from embeddings.config import CSV_PATH_district as DISTRICT_CSV
from embeddings.upload_embeddings import upload

def run_district():
    upload(csv_path=DISTRICT_CSV)

if __name__ == "__main__":
    run_district()
