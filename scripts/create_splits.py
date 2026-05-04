import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_INPUT = Path("data/raw/final/mh_signal_data_w-concern-intent.csv")
DEFAULT_OUTPUT_DIR = Path("data/splits")
DEFAULT_TEST_SIZE = 0.30
DEFAULT_VAL_SIZE = 0.50
DEFAULT_RANDOM_STATE = 42

COL_POST = "Post"
COL_TAG = "Tag"
COL_CONCERN = "Concern_Level"


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[[COL_POST, COL_TAG, COL_CONCERN]].copy()
    df[COL_TAG] = df[COL_TAG].fillna("")
    df = df.dropna(subset=[COL_POST, COL_CONCERN])
    df[COL_CONCERN] = df[COL_CONCERN].astype(str).str.strip().str.lower()
    return df


def split_data(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    train, temp = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df[COL_CONCERN],
    )
    val, test = train_test_split(
        temp,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=temp[COL_CONCERN],
    )
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def print_split_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(name, split[COL_CONCERN].value_counts(normalize=True).round(3).to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified concern-level train/val/test splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV file path.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output splits directory.")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, help="Test split proportion.")
    parser.add_argument("--val_size", type=float, default=DEFAULT_VAL_SIZE, help="Validation split proportion from the remaining data.")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for reproducible splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    train, val, test = split_data(df, args.test_size, args.val_size, args.random_state)
    save_splits(train, val, test, args.output_dir)
    print_split_stats(train, val, test)
    print(f"Splits created at {args.output_dir}/")


if __name__ == "__main__":
    main()
