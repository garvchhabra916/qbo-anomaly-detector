"""Measure detection quality against synthetic ground truth."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

MODES = [("rules_only", True, False), ("ml_only", False, True), ("combined", True, True)]


def evaluate(scored: pd.DataFrame, truth: pd.DataFrame, threshold: float = 0.5) -> dict:
    labeled = scored.merge(truth, on="txn_id", how="left")
    is_anom = labeled["anomaly_type"].notna()
    flagged = labeled["risk_score"] >= threshold

    tp = int((flagged & is_anom).sum())
    fp = int((flagged & ~is_anom).sum())
    fn = int((~flagged & is_anom).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    k = int(is_anom.sum())
    top_k = labeled.nlargest(k, "risk_score")
    per_type = (labeled[is_anom].assign(caught=flagged[is_anom])
                .groupby("anomaly_type")["caught"].agg(["sum", "size"]))
    per_type["recall"] = per_type["sum"] / per_type["size"]

    return {
        "transactions": len(labeled), "anomalies": k, "flagged": int(flagged.sum()),
        "precision": precision, "recall": recall, "f1": f1,
        f"precision_at_{k}": float(top_k["anomaly_type"].notna().mean()),
        "per_type_recall": per_type["recall"].round(3).to_dict(),
    }


def benchmark(seeds: range = range(1, 21), threshold: float = 0.5) -> pd.DataFrame:
    """Average metrics over many synthetic companies so one lucky seed can't flatter the model."""
    from . import ingest, synth
    from .score import scan

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            payload, truth = synth.generate(seed=seed)
            path = Path(tmp) / f"{seed}.json"
            path.write_text(json.dumps(payload))
            df = ingest.load(path)
            for mode, use_rules, use_ml in MODES:
                e = evaluate(scan(df, use_rules=use_rules, use_ml=use_ml), truth, threshold)
                rows.append({"mode": mode, "precision": e["precision"], "recall": e["recall"], "f1": e["f1"],
                             **{f"recall:{k}": v for k, v in e["per_type_recall"].items()}})
    return pd.DataFrame(rows).groupby("mode", sort=False).agg(["mean", "std"]).round(3)
