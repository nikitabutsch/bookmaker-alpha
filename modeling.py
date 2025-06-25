"""
Train a model that predicts the 2-to-3-day correction following high-surprise matches.
Runs CatBoost on surprise factor, Day-1 move and league context.
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import config


def _load_data(path=f"{config.RESULTS_DIR}/alpha_dataset.csv", threshold: float = 0.7) -> pd.DataFrame:
    """Return only high-surprise games and derive correction_return if missing."""
    df = pd.read_csv(path, parse_dates=["match_date"])

    if "correction_return" not in df.columns and {
        "next_day_return", "three_day_return"
    }.issubset(df.columns):
        df["correction_return"] = (1 + df["three_day_return"]) / (1 + df["next_day_return"]) - 1

    df = df.dropna(subset=["surprise_factor", "next_day_return", "correction_return"])
    return df[df["surprise_factor"] > threshold].copy()


def main():
    data = _load_data()
    print(f"High-surprise sample: {len(data)} matches")

    feature_cols = [
        "surprise_factor",
        "next_day_return",
        "bvb_home",
        "bvb_away",
        "is_bundesliga",
        "is_champions_league",
        "is_europa_league",
        "is_domestic_cup",
        "is_friendly",
    ]

    X, y = data[feature_cols].fillna(0), data["correction_return"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE
    )
    print(f"Train n = {len(X_train)}, Test n = {len(X_test)}")

    model = CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=config.RANDOM_STATE,
        verbose=False,
    )

    print("🧠  Training CatBoost regressor…")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    direction_acc = (np.sign(preds) == np.sign(y_test)).mean()

    print(f"\n📋  Hold-out metrics:\n  RMSE = {rmse:.4f}\n  R²   = {r2:.3f}\n  Sign accuracy = {direction_acc:.2%}")

    print("\nFeature importances:")
    for f, imp in sorted(zip(feature_cols, model.get_feature_importance()), key=lambda x: x[1], reverse=True):
        print(f"  {f:22s}: {imp:5.1f}")


if __name__ == "__main__":
    main() 