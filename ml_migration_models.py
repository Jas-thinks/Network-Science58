from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_clean" / "final_migration_dataset.csv"
OUT = BASE / "ml_outputs"
OUT.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Target
    y = df["migrants"].astype(float)

    # Candidate feature columns (origin, destination, and pair)
    candidates = []
    prefixes = [
        "gdp_", "population_", "unemployment_", "education_", "conflict_", "climate_",
        "visa_", "remittances_", "stability_", "internet_",
        "distance_", "common_language", "colonial_tie",
    ]

    for col in df.columns:
        if col == "migrants":
            continue
        if any(col.startswith(p) for p in prefixes) or col in {"distance_km", "common_language", "colonial_tie"}:
            candidates.append(col)

    if not candidates and "year" in df.columns:
        candidates.append("year")

    X = df[candidates].copy()
    return X, y


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols)],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {"r2": float(r2), "rmse": float(rmse)}


def main() -> None:
    df = load_data(DATA)
    X, y = select_features(df)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)

        # Feature importance (where possible)
        if name in {"linear", "ridge", "lasso"}:
            coefs = pipe.named_steps["model"].coef_
            feats = numeric_cols
            importance = pd.DataFrame({"feature": feats, "importance": coefs}).sort_values(
                "importance", key=lambda s: s.abs(), ascending=False
            )
            importance.to_csv(OUT / f"feature_importance_{name}.csv", index=False)
        elif name in {"random_forest", "xgboost"}:
            importances = pipe.named_steps["model"].feature_importances_
            feats = numeric_cols
            importance = pd.DataFrame({"feature": feats, "importance": importances}).sort_values(
                "importance", ascending=False
            )
            importance.to_csv(OUT / f"feature_importance_{name}.csv", index=False)

    # Save model results
    with open(OUT / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary insights
    best_model = max(results.items(), key=lambda x: x[1]["r2"])
    print("Model comparison:")
    for k, v in results.items():
        print(f"{k}: R2={v['r2']:.4f}, RMSE={v['rmse']:.2f}")
    print(f"Best model: {best_model[0]}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
