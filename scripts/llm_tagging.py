import argparse
from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


# --- Configuration ---
INPUT_FILE = "data/llm_taged/mh_signal_data_w-intent.csv"
OUTPUT_DATASET_FILE = "data/llm_taged/full_dataset_tagged.csv"
OUTPUT_EVAL_FILE = "data/llm_taged/evaluation_report.csv"
BATCH_SIZE = 16
MODEL_NAME = "facebook/bart-large-mnli"


# --- Labels ---
ALL_LABELS = [
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

SEMANTIC_LABELS = [
    ("A person is in critical risk of suicide or self-harm", "Critical Risk"),
    ("A person is expressing feelings of depression, anxiety, or general distress", "Mental Distress"),
    ("A person is describing a negative coping mechanism (e.g., substance abuse, self-isolation)", "Maladaptive Coping"),
    ("A person is describing a positive coping mechanism (e.g., exercise, journaling, meditation)", "Positive Coping"),
    ("A person is asking for help, advice, or resources", "Seeking Help"),
    ("A person is sharing an update on their treatment, medication, or therapy", "Progress Update"),
    ("A person is sharing a log of their current mood or feelings", "Mood Tracking"),
    ("A person is identifying a specific reason for their distress (e.g., job, relationship, school)", "Cause of Distress"),
]
SEMANTIC_LABEL_NAMES = [label for _, label in SEMANTIC_LABELS]


def parse_human_tags(raw_tags: str) -> list:
    """Convert a comma/semicolon-separated tag string into canonical tag labels."""
    if not isinstance(raw_tags, str) or not raw_tags.strip():
        return []

    normalized_map = {
        **{label.lower(): label for label in ALL_LABELS},
        **{label: label for label in ALL_LABELS},
        "cause of distress": "Cause of Distress",
        "causes of distress": "Cause of Distress",
        "progress update": "Progress Update",
        "progress update. cause of distress": "Progress Update",
    }

    cleaned_labels = {
        normalized_map[tag.strip().strip('"').lower()]
        for tag in re.split(r"[;,]", raw_tags)
        if tag.strip().strip('"').lower() in normalized_map
    }

    return list(cleaned_labels) if cleaned_labels else ["Miscellaneous"]


def get_model_scores(texts: list, classifier, batch_size: int) -> list:
    """Run zero-shot classification and return raw label score dictionaries."""
    print(f"\nGetting model scores for {len(texts)} posts...")

    descriptions = [description for description, _ in SEMANTIC_LABELS]
    description_to_label = {description: label for description, label in SEMANTIC_LABELS}

    scores = []
    for output in tqdm(
        classifier(texts, candidate_labels=descriptions, multi_label=True, batch_size=batch_size),
        total=len(texts),
        desc="Classifying posts",
    ):
        row_scores = {label: 0.0 for label in SEMANTIC_LABEL_NAMES}
        for predicted_label, score in zip(output["labels"], output["scores"]):
            row_scores[description_to_label[predicted_label]] = score
        scores.append(row_scores)

    return scores


def find_optimal_thresholds(y_true_bin, y_pred_scores_df):
    """Find per-label thresholds that maximize F1 score."""
    print("\n--- Finding Optimal Thresholds ---")
    optimal_thresholds = {}

    for label in SEMANTIC_LABEL_NAMES:
        y_true_col = y_true_bin[label]
        y_prob_col = y_pred_scores_df[label]

        precisions, recalls, thresholds = precision_recall_curve(y_true_col, y_prob_col)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

        best_index = int(np.nanargmax(f1_scores))
        optimal_thresholds[label] = float(thresholds[best_index])
        print(f"Optimal threshold for {label:<20}: {optimal_thresholds[label]:.4f} (F1: {f1_scores[best_index]:.4f})")

    return optimal_thresholds


def apply_thresholds(y_pred_scores_df, thresholds):
    """Convert raw score dataframe into final label lists using thresholds."""
    print("\nApplying optimal thresholds to get final tags...")
    results = []
    for row in y_pred_scores_df.to_dict("records"):
        tags = [label for label, score in row.items() if score >= thresholds.get(label, 0.5)]
        results.append(tags or ["Miscellaneous"])
    return results


def create_classifier() -> tuple:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if device == -1:
        print("WARNING: No GPU detected. This will be very slow.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
    return tokenizer, classifier


def truncate_text(tokenizer, text: str) -> str:
    tokenized = tokenizer(text, truncation=True, max_length=tokenizer.model_max_length - 2)
    return tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)


def build_final_dataframe(labeled_df, unlabeled_df):
    if labeled_df is not None and not labeled_df.empty and unlabeled_df is not None and not unlabeled_df.empty:
        return pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    return labeled_df if labeled_df is not None and not labeled_df.empty else unlabeled_df


def main(input_path, output_dataset, output_evaluation, batch_size):
    print("Setting up model pipeline...")
    warnings.filterwarnings("ignore", ".*Using a pipeline without specifying a model name.*")

    tokenizer, classifier = create_classifier()

    raw_df = pd.read_csv(input_path)
    if "Post" in raw_df.columns and "Text" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"Post": "Text"})

    raw_df["Text"] = raw_df["Text"].fillna("").astype(str)
    raw_df["Tag"] = raw_df["Tag"].fillna("").astype(str)
    raw_df["Truncated_Text"] = raw_df["Text"].apply(lambda text: truncate_text(tokenizer, text))

    labeled_df = raw_df[raw_df["Tag"] != ""].copy()
    unlabeled_df = raw_df[raw_df["Tag"] == ""].copy()

    print(f"Found {len(labeled_df)} manually labeled posts.")
    print(f"Found {len(unlabeled_df)} unlabeled posts to tag.")

    labeled_output = None
    unlabeled_output = None
    optimal_thresholds = None

    if not labeled_df.empty:
        print("\n--- Evaluating Model on Manually tagged Set ---")
        labeled_df["Human_Tags"] = labeled_df["Tag"].apply(parse_human_tags)
        mlb = MultiLabelBinarizer(classes=ALL_LABELS)
        y_true = mlb.fit_transform(labeled_df["Human_Tags"])
        y_true_bin_df = pd.DataFrame(y_true, columns=mlb.classes_)

        y_pred_scores_df = pd.DataFrame(get_model_scores(labeled_df["Truncated_Text"].tolist(), classifier, batch_size))
        optimal_thresholds = find_optimal_thresholds(y_true_bin_df, y_pred_scores_df)

        labeled_df["Model_Tags"] = apply_thresholds(y_pred_scores_df, optimal_thresholds)
        labeled_df["Final_Tags"] = labeled_df["Human_Tags"]
        labeled_df["Tag_Source"] = "Human_Gold"

        labeled_df[["Text", "Human_Tags", "Model_Tags"]].to_csv(output_evaluation, index=False, encoding="utf-8")
        print(f"Saved side-by-side evaluation to: {output_evaluation}")
        print("\n--- Final Report (with Optimal Thresholds) ---")
        print(classification_report(y_true, mlb.transform(labeled_df["Model_Tags"]), target_names=mlb.classes_, zero_division=0))
        labeled_output = labeled_df

    if not unlabeled_df.empty:
        print("\n--- Tagging Unlabeled Set ---")
        y_pred_scores_df = pd.DataFrame(get_model_scores(unlabeled_df["Truncated_Text"].tolist(), classifier, batch_size))
        if optimal_thresholds is None:
            optimal_thresholds = {label: 0.5 for label in SEMANTIC_LABEL_NAMES}
        unlabeled_df["Final_Tags"] = apply_thresholds(y_pred_scores_df, optimal_thresholds)
        unlabeled_df["Tag_Source"] = "Model_Optimal"
        unlabeled_output = unlabeled_df

    final_df = build_final_dataframe(labeled_output, unlabeled_output)
    if final_df is None:
        final_df = pd.DataFrame(columns=raw_df.columns.tolist() + ["Final_Tags", "Tag_Source"])

    final_cols = ["Text", "Tag", "Final_Tags", "Tag_Source"]
    extra_cols = [column for column in final_df.columns if column not in final_cols and column not in ["Truncated_Text", "Human_Tags", "Model_Tags"]]
    final_df = final_df[extra_cols + final_cols]

    final_df.to_csv(output_dataset, index=False, encoding="utf-8")
    print(f"\nSaved full {len(final_df)}-post tagged dataset to: {output_dataset}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag dataset with zero-shot semantic intent labels.")
    parser.add_argument("--input_path", default=INPUT_FILE)
    parser.add_argument("--output_dataset", default=OUTPUT_DATASET_FILE)
    parser.add_argument("--output_evaluation", default=OUTPUT_EVAL_FILE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    Path(args.output_dataset).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_evaluation).parent.mkdir(parents=True, exist_ok=True)

    main(args.input_path, args.output_dataset, args.output_evaluation, args.batch_size)
