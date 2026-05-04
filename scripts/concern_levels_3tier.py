import argparse
from pathlib import Path

import pandas as pd

ALL_TAGS = {
    "critical risk",
    "mental distress",
    "maladaptive coping",
    "positive coping",
    "seeking help",
    "progress update",
    "mood tracking",
    "cause of distress",
}


def parse_tags(tag_string: str) -> list[str]:
    return [token.strip() for token in str(tag_string).split(",") if token.strip()]


def determine_concern_level(tags_string: str) -> tuple[str, str, str]:
    normalized_tags = {tag.lower() for tag in parse_tags(tags_string)}
    normalized_tags = {tag for tag in normalized_tags if tag in ALL_TAGS}

    if "critical risk" in normalized_tags:
        notes = []
        if "maladaptive coping" in normalized_tags:
            notes.append("with Maladaptive Coping")
        if "progress update" in normalized_tags:
            notes.append("with Progress Update")
        if "seeking help" in normalized_tags:
            notes.append("and Seeking Help")
        reason = "Critical Risk"
        return "High", reason, "; ".join(notes) if notes else ""

    if normalized_tags & {"seeking help", "maladaptive coping", "progress update"}:
        reason = ", ".join(sorted(x.title() for x in normalized_tags & {"seeking help", "maladaptive coping", "progress update"}))
        notes = ", ".join(sorted(x.title() for x in normalized_tags & {"mental distress", "cause of distress"}))
        return "Medium", reason, notes

    if normalized_tags & {"mental distress", "cause of distress"}:
        reason = ", ".join(sorted(x.title() for x in normalized_tags & {"mental distress", "cause of distress"}))
        notes = ", ".join(sorted(x.title() for x in normalized_tags & {"positive coping", "mood tracking"}))
        return "Medium", reason, notes

    reason = ", ".join(sorted(x.title() for x in normalized_tags)) or "No higher-severity tags"
    return "Low", reason, ""


def annotate_concern(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    tag_column = "Tags" if "Tags" in df.columns else df.columns[-1]

    results = [determine_concern_level(tag_string) for tag_string in df[tag_column].astype(str)]
    concern_levels, _, _ = zip(*results) if results else ([], [], [])

    df.insert(len(df.columns), "Concern_Level", concern_levels)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign 3-tier concern levels from tag strings.")
    parser.add_argument("--input", type=Path, default=Path("this_tagged.csv"), help="Input CSV file path.")
    parser.add_argument("--output", type=Path, default=Path("this_with_concern.csv"), help="Output CSV file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotate_concern(args.input, args.output)


if __name__ == "__main__":
    main()
