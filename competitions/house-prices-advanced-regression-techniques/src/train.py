from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_data, prepare_features, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main():
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"

    train_df, test_df = load_data(train_path, test_path)
    X, X_test, y = prepare_features(train_df, test_df)

    preprocessor = build_preprocessor(X)
    model = XGBRegressor(objective="reg:squarederror", random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    valid_pred = pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, valid_pred, squared=False)
    mae = mean_absolute_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R2: {r2:.4f}")

    test_pred = pipeline.predict(X_test)
    submission = pd.DataFrame({
        "Id": pd.read_csv(test_path)["Id"],
        "SalePrice": test_pred,
    })

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)
    print("Submission saved to:", SUBMISSIONS_DIR / "submission.csv")


if __name__ == "__main__":
    main()
