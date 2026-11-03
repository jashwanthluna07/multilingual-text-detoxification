#!/usr/bin/env python3
"""
Shared evaluation pipeline for multilingual text detoxification.

Expected test CSV columns:
    id, toxic_sentence, neutral_sentence

Prediction CSV columns:
    id, generated_sentence
The prediction text column may also be named prediction, output, or generated_text.

Important:
- STA is the classifier's probability for the label mapped to "neutral".
- SIM is LaBSE cosine similarity between toxic input and generated output.
- ChrF is sentence-level SacreBLEU chrF, normalized from [0, 100] to [0, 1].
- The COLING 2025 paper's published English values are consistent with
  Joint = mean(STA_i * SIM_i * ChrF_i), NOT the cube root of that product.
  Example: human reference 0.864 * 0.820 * 1.000 ~= 0.708, reported as 0.711.
- Failed/missing generations remain in the denominator and receive zero for all
  four per-example metrics. This avoids silently rewarding a model for failures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TOXICITY_MODEL = "textdetox/xlmr-large-toxicity-classifier"
DEFAULT_SIM_MODEL = "sentence-transformers/LaBSE"
PREDICTION_COLUMNS = ["generated_sentence", "prediction", "output", "generated_text", "generated"]
ID_COLUMNS = ["id", "example_id", "sentence_id"]


def first_existing_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not identify {description} column. Tried {candidates}; "
        f"available columns: {list(df.columns)}"
    )


def normalize_text(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<failed>", "[failed]"}:
        return None
    return text


def load_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = first_existing_column(df, ID_COLUMNS, "test example ID")
    required = ["toxic_sentence", "neutral_sentence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV missing required columns {missing}. Found: {list(df.columns)}")
    df = df.rename(columns={id_col: "id"}).copy()
    if df["id"].isna().any():
        raise ValueError("Test CSV contains missing IDs.")
    df["id"] = df["id"].astype(str)
    if df["id"].duplicated().any():
        duplicates = df.loc[df["id"].duplicated(), "id"].tolist()[:10]
        raise ValueError(f"Test CSV has duplicate IDs, examples: {duplicates}")
    df["toxic_sentence"] = df["toxic_sentence"].map(normalize_text)
    df["neutral_sentence"] = df["neutral_sentence"].map(normalize_text)
    if df[["toxic_sentence", "neutral_sentence"]].isna().any().any():
        raise ValueError("Test CSV contains empty toxic_sentence or neutral_sentence.")
    return df[["id", "toxic_sentence", "neutral_sentence"]]


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = first_existing_column(df, ID_COLUMNS, "prediction example ID")
    pred_col = first_existing_column(df, PREDICTION_COLUMNS, "generated prediction")
    df = df.rename(columns={id_col: "id", pred_col: "generated_sentence"}).copy()
    if df["id"].isna().any():
        raise ValueError(f"Prediction file {path} contains missing IDs.")
    df["id"] = df["id"].astype(str)
    if df["id"].duplicated().any():
        duplicates = df.loc[df["id"].duplicated(), "id"].tolist()[:10]
        raise ValueError(f"Prediction file {path} has duplicate IDs, examples: {duplicates}")
    df["generated_sentence"] = df["generated_sentence"].map(normalize_text)
    return df[["id", "generated_sentence"]]


def load_toxicity_model(model_name: str, device: str):
    print(f"Loading toxicity classifier: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    print("Classifier id2label:", id2label, flush=True)
    neutral_ids = [i for i, label in id2label.items() if "neutral" in label.lower() or "non-toxic" in label.lower() or "nontoxic" in label.lower()]
    toxic_ids = [i for i, label in id2label.items() if "toxic" in label.lower() and i not in neutral_ids]
    if not neutral_ids:
        raise ValueError(
            "Could not infer the non-toxic label from id2label. "
            f"Refusing to guess. id2label={id2label}"
        )
    if len(neutral_ids) > 1:
        print(f"Warning: multiple possible neutral labels {neutral_ids}; using first.", file=sys.stderr)
    neutral_id = neutral_ids[0]
    return tokenizer, model, neutral_id, id2label


@torch.inference_mode()
def predict_sta(texts: list[str], tokenizer, model, neutral_id: int, device: str, batch_size: int, max_length: int):
    """Return neutral-class probabilities, with batches to bound GPU memory."""
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, neutral_id]
        outputs.extend(probs.detach().cpu().tolist())
    return outputs


def load_similarity_model(model_name: str, device: str):
    print(f"Loading LaBSE model: {model_name}", flush=True)
    return SentenceTransformer(model_name, device=device)


def cosine_similarities(inputs: list[str], outputs: list[str], model, batch_size: int):
    """Compute pairwise cosine similarities; embeddings are normalized."""
    e1 = model.encode(
        inputs, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    e2 = model.encode(
        outputs, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    vals = np.sum(e1 * e2, axis=1)
    # Preserve cosine's natural [-1, 1] range; do not rescale without paper/code support.
    return np.clip(vals, -1.0, 1.0).tolist()


def sentence_chrf_scores(hypotheses: list[str], references: list[str]) -> list[float]:
    # SacreBLEU defaults: char_order=6, word_order=0 (chrF, not chrF++),
    # beta=2, whitespace=False (remove whitespace), eps_smoothing=False.
    metric = CHRF(char_order=6, word_order=0, beta=2, whitespace=False, eps_smoothing=False)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        # sentence_score expects a list of references and returns score on [0, 100].
        scores.append(float(metric.sentence_score(hyp, [ref]).score) / 100.0)
    return scores


def evaluate_one(
    test_df: pd.DataFrame,
    prediction_path: str,
    model_name: str,
    output_dir: Path,
    toxicity_model_name: str,
    similarity_model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
):
    pred_df = load_predictions(prediction_path)
    unknown = sorted(set(pred_df["id"]) - set(test_df["id"]))
    if unknown:
        raise ValueError(
            f"{prediction_path} has {len(unknown)} IDs not present in test set; "
            f"examples: {unknown[:10]}"
        )

    merged = test_df.merge(pred_df, on="id", how="left", validate="one_to_one")
    merged["generation_failed"] = merged["generated_sentence"].isna()
    n_failed = int(merged["generation_failed"].sum())
    if n_failed:
        print(f"[{model_name}] {n_failed}/{len(merged)} missing/failed generations; scoring them as zero.", flush=True)

    valid = ~merged["generation_failed"]
    merged["STA"] = 0.0
    merged["SIM"] = 0.0
    merged["ChrF1"] = 0.0

    if valid.any():
        valid_texts = merged.loc[valid, "generated_sentence"].tolist()
        tox_tok, tox_model, neutral_id, label_map = load_toxicity_model(toxicity_model_name, device)
        merged.loc[valid, "STA"] = predict_sta(
            valid_texts, tox_tok, tox_model, neutral_id, device, batch_size, max_length
        )
        del tox_model, tox_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sim_model = load_similarity_model(similarity_model_name, device)
        sims = cosine_similarities(
            merged.loc[valid, "toxic_sentence"].tolist(),
            valid_texts, sim_model, batch_size
        )
        merged.loc[valid, "SIM"] = sims
        del sim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chrf = sentence_chrf_scores(
            valid_texts, merged.loc[valid, "neutral_sentence"].tolist()
        )
        merged.loc[valid, "ChrF1"] = chrf
    else:
        label_map = {}

    # Official paper values are consistent with mean of the per-sentence product,
    # not a cube root. See evaluation_config.txt for explanation.
    merged["Joint"] = merged["STA"] * merged["SIM"] * merged["ChrF1"]
    merged.insert(0, "model_name", model_name)
    merged["prediction_file"] = str(prediction_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    per_path = output_dir / f"{safe_name}_per_example_metrics.csv"
    merged.to_csv(per_path, index=False)

    summary = {
        "model_name": model_name,
        "STA": float(merged["STA"].mean()),
        "SIM": float(merged["SIM"].mean()),
        "ChrF1": float(merged["ChrF1"].mean()),
        "Joint": float(merged["Joint"].mean()),
        "num_examples": int(len(merged)),
        "failed_generations": n_failed,
    }
    return summary, merged, label_map


def update_consolidated(summary: dict, output_dir: Path):
    path = output_dir / "evaluation_results.csv"
    new = pd.DataFrame([summary])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["model_name"] != summary["model_name"]]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values("Joint", ascending=False, na_position="last")
    new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a text-detoxification prediction CSV.")
    parser.add_argument("--test", default="data/test.csv", help="Test CSV with id,toxic_sentence,neutral_sentence")
    parser.add_argument("--predictions", required=True, help="Prediction CSV with ID and generated sentence")
    parser.add_argument("--model-name", required=True, help="Model label, e.g. mbart_400")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--toxicity-model", default=DEFAULT_TOXICITY_MODEL)
    parser.add_argument("--similarity-model", default=DEFAULT_SIM_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    test_df = load_test(args.test)
    summary, per_example, label_map = evaluate_one(
        test_df=test_df,
        prediction_path=args.predictions,
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        toxicity_model_name=args.toxicity_model,
        similarity_model_name=args.similarity_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    update_consolidated(summary, Path(args.output_dir))
    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Per-example metrics: {Path(args.output_dir) / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in args.model_name) + '_per_example_metrics.csv')}")
    print(f"Consolidated table: {Path(args.output_dir) / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()
