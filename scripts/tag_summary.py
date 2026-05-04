import argparse
import os
from pathlib import Path
from collections import Counter

import pandas as pd
import yaml

CANONICAL_TAGS = [
    "Critical Risk",
    "Mental Distress",
    "Maladaptive Coping",
    "Positive Coping",
    "Seeking Help",
    "Progress Update",
    "Mood Tracking",
    "Cause of Distress",
    "Miscellaneous",
]

TAG_ALIAS = {
    "critical risk": "Critical Risk",
    "mental distress": "Mental Distress",
    "maladaptive coping": "Maladaptive Coping",
    "positive coping": "Positive Coping",
    "seeking help": "Seeking Help",
    "progress update": "Progress Update",
    "mood tracking": "Mood Tracking",
    "cause of distress": "Cause of Distress",
    "cause of distress.": "Cause of Distress",
    "misc": "Miscellaneous",
    "miscellaneous": "Miscellaneous",
}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_tag(raw_tag: str) -> str | None:
    normalized = " ".join(str(raw_tag).strip().split()).lower()
    return TAG_ALIAS.get(normalized)


def summarize_tags(tagged_csv: str, output_summary_csv: str) -> pd.DataFrame:
    if not Path(tagged_csv).exists():
        raise FileNotFoundError(f"File not found: {tagged_csv}")

    df = pd.read_csv(tagged_csv)
    if "Tag" not in df.columns:
        raise KeyError("Column 'Tag' not found in CSV.")

    counts = Counter()
    unknown = Counter()

    for raw_tags in df["Tag"].fillna(""):
        label_strings = [label.strip() for label in str(raw_tags).split(",") if label.strip()]
        seen = set()
        for label in label_strings:
            canonical_label = normalize_tag(label)
            if canonical_label is None:
                unknown[label.strip()] += 1
                continue
            if canonical_label in CANONICAL_TAGS and canonical_label not in seen:
                counts[canonical_label] += 1
                seen.add(canonical_label)

    total = sum(counts.get(tag, 0) for tag in CANONICAL_TAGS)
    rows = []
    for tag in CANONICAL_TAGS:
        count = counts.get(tag, 0)
        percent = round(100 * count / total, 2) if total else 0.0
        rows.append({"Tag": tag, "Count": count, "Percent": percent})

    summary_df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    Path(output_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8")
    print(summary_df.to_string(index=False))

    if unknown:
        unknown_df = pd.DataFrame(sorted(unknown.items(), key=lambda item: -item[1]), columns=["UnknownTag", "Count"])
        unknown_path = Path(output_summary_csv).with_name("unknown_tags.csv")
        unknown_df.to_csv(unknown_path, index=False, encoding="utf-8")
        print(f"\nUnknown tags saved to: {unknown_path}")

    print(f"\nSaved to: {output_summary_csv}")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tag distributions from a tagged CSV file.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--output", default="data/processed/tags_summary.csv", help="Output summary CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tagged_csv = cfg["paths"]["tagged_data"]
    summarize_tags(tagged_csv, args.output)


if __name__ == "__main__":
    main()
