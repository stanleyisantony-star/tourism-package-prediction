import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import upload_file
from sklearn.model_selection import train_test_split

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO_ID = "HfStan/tourism-wellness-dataset"  # TODO: update
TARGET_COL = "ProdTaken"


def clean_data(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in ["Gender", "Occupation", "TypeofContact", "ProductPitched", "MaritalStatus", "Designation"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=[TARGET_COL])

    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    ds = load_dataset(DATASET_REPO_ID, data_files={"full": "tourism.csv"})["full"]
    df = ds.to_pandas()

    df = clean_data(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    os.makedirs("data/processed", exist_ok=True)
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

if __name__ == "__main__":
    main()
