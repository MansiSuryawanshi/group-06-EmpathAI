# models/helper.py
import ast
import math
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

# --- Canonical Tag Mapping ---
CANONICAL = {
    "critical risk": "Critical Risk", 
    "mental distress": "Mental Distress",
    "maladaptive coping": "Maladaptive Coping", 
    "positive coping": "Positive Coping",
    "seeking help": "Seeking Help", 
    "progress update": "Progress Update",
    "mood tracking": "Mood Tracking", 
    "cause of distress": "Cause of Distress",
    "miscellaneous": "Miscellaneous",
}
CANON_KEYS = set(CANONICAL.values())

# --- Utilities ---
def set_seed(s: int):
    """Set all random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def load_yaml(p: str):
    """Load YAML configuration from a given path."""
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    """Ensure a directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p

# --- Data Processing Helpers ---
def _normalize_tag_to_canonical(raw_tag: str) -> str | None:
    """Normalize a raw tag string into the canonical label mapping."""
    text = str(raw_tag).strip().lower()
    text = re.sub(r"\.$", "", text)
    text = text.replace("causes of distress", "cause of distress")
    text = text.replace("progress update.", "progress update")
    return CANONICAL.get(text)


def _extract_tag_values(raw_value):
    """Parse a raw tag field into a list of individual tag strings."""
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []

    text = str(raw_value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return [str(item) for item in parsed if item is not None]
        except (ValueError, SyntaxError):
            return []

    return [value.strip() for value in re.split(r"[;,]", text) if value.strip()]


def _normalize_tag_list(raw_value):
    """Convert a raw tag field into a canonical list of tags."""
    values = _extract_tag_values(raw_value)
    normalized = []
    seen = set()
    for value in values:
        canonical = _normalize_tag_to_canonical(value)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized or ["Miscellaneous"]


def read_and_process_data(p: Path):
    """Read the raw CSV and normalize the tag column to canonical tag lists."""
    df = pd.read_csv(p)

    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' or 'Text' column missing in {p}")

    if "Final_Tags" in df.columns:
        tag_column = "Final_Tags"
    elif "Tag" in df.columns:
        tag_column = "Tag"
    else:
        raise ValueError(f"'Tag' or 'Final_Tags' column missing in {p}")

    df["Post"] = df["Post"].fillna("").astype(str)
    df["TagsList"] = df[tag_column].apply(_normalize_tag_list)
    return df[["Post", "TagsList"]]


def read_split_csv(p: Path):
    """Read a split CSV and normalize the tag column into canonical tag lists."""
    df = pd.read_csv(p)

    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    tag_col = None
    if "Tag" in df.columns:
        tag_col = "Tag"
    elif "Final_Tags" in df.columns:
        tag_col = "Final_Tags"
    else:
        raise ValueError(f"'Tag' or 'Final_Tags' column missing in {p}")

    df["Post"] = df["Post"].fillna("").astype(str)
    df["TagsList"] = df[tag_col].apply(_normalize_tag_list)
    return df[["Post", "TagsList"]]

def prob_to_tags(prob_row, threshold, names):
    """Convert probability vector to tag string based on threshold."""
    idx = np.where(prob_row >= threshold)[0].tolist()
    if not idx:
        idx = [int(np.argmax(prob_row))]
    return ", ".join([names[i] for i in idx])

def _normalize_concern_level(x: str) -> str | None:
    """
    Normalizes the 'Concern_Level' string.
    (Kept here as it's specific to this task)
    """
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    t = re.sub(r"[.\s]+$", "", t)
    if t in {"low", "medium", "high"}:
        return t
    if t in {"med", "mid"}:
        return "medium"
    return None


def read_concern_split_csv(p: Path):
    """
    Reads the CSV and prepares it for the Concern Level (single-label) task.
    (Kept here as it's different from the helper.py version which reads 'Tag')
    """
    df = pd.read_csv(p)
    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    if "Concern_Level" not in df.columns:
        raise ValueError(f"'Concern_Level' column missing in {p}")
    df["Post"] = df["Post"].fillna("").astype(str)
    df["Concern_Level"] = df["Concern_Level"].apply(_normalize_concern_level)
    df = df.dropna(subset=["Concern_Level"]).reset_index(drop=True)
    return df[["Post", "Concern_Level"]]
