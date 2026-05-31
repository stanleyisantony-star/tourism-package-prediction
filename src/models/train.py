import os
import joblib
import pandas as pd
from datasets import load_dataset
from huggingface_hub import create_repo, upload_file
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO_ID = "HfStan/tourism-wellness-dataset"  # TODO: update
MODEL_REPO_ID = "HfStan/tourism-wellness-model"      # TODO: update
TARGET_COL = "ProdTaken"


def load_splits():
    train_ds = load_dataset(DATASET_REPO_ID, data_files={"train": "train.csv"})["train"]
    test_ds = load_dataset(DATASET_REPO_ID, data_files={"test": "test.csv"})["test"]
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()
    return train_df, test_df


def build_model():
    categorical_cols = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
    ]

    numeric_cols = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "MonthlyIncome",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = GradientBoostingClassifier(random_state=42)

    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    return model


def train_and_evaluate():
    train_df, test_df = load_splits()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    model = build_model()

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 4],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)

    os.makedirs("experiments", exist_ok=True)
    log_path = os.path.join("experiments", "experiment_log.csv")
    import csv

    row = {**grid.best_params_, "auc": auc, "f1": f1, "accuracy": acc}
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return model_path


def register_model(model_path: str):
    create_repo(
        repo_id=MODEL_REPO_ID,
        token=HF_TOKEN,
        repo_type="model",
        exist_ok=True,
    )

    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
    )


def main():
    model_path = train_and_evaluate()
    register_model(model_path)


if __name__ == "__main__":
    main()
