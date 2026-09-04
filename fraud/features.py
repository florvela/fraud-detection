"""Features: deriva hour/age, arma la tabla y hace el split estratificado train/val/test"""

from __future__ import annotations

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from fraud.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

TARGET = "is_fraud"
NUMERIC_FEATURES = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "hour", "age"]
CATEGORICAL_FEATURES = ["category", "gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva hour/age desde los campos de texto y devuelve solo FEATURES + TARGET"""
    df = df.copy()
    df["hour"] = df["trans_date_trans_time"].str.slice(11, 13).astype(int)
    df["age"] = (
        df["trans_date_trans_time"].str.slice(0, 4).astype(int)
        - df["dob"].str.slice(0, 4).astype(int)
    )
    return df[FEATURES + [TARGET]]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado en train/val/test (preserva la proporción real de fraude)"""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, stratify=train_val[TARGET]
    )
    return train, val, test


@app.command()
def main(seed: int = typer.Option(42, help="Semilla del split")) -> None:
    """Lee data/raw, arma features, splitea y guarda en data/processed/"""
    df = pd.read_parquet(RAW_DATA_DIR / "fraud_sample.parquet")
    features = build_features(df)
    train, val, test = split_data(features, seed=seed)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED_DATA_DIR / f"{name}.parquet", index=False)
        logger.info(f"{name:5s}: {len(part):>7,} filas | fraude {part[TARGET].mean():.2%}")

    logger.success("Splits guardados en data/processed/")


if __name__ == "__main__":
    app()
