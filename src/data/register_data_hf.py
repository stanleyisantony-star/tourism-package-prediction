import os
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO_ID = "HfStan/tourism-wellness-dataset"  # TODO: update

def main():
    assert HF_TOKEN is not None, "HF_TOKEN env var must be set"

    HfFolder.save_token(HF_TOKEN)
    api = HfApi()

    create_repo(
        repo_id=DATASET_REPO_ID,
        token=HF_TOKEN,
        repo_type="dataset",
        exist_ok=True,
    )

    local_csv_path = os.path.join("data", "raw", "tourism.csv")
    assert os.path.exists(local_csv_path), f"{local_csv_path} not found"

    upload_file(
        path_or_fileobj=local_csv_path,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

if __name__ == "__main__":
    main()
