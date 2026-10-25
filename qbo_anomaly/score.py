"""Combine rules and the ML score into one ranked, explained risk score."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .ml import isolation_scores
from .rules import RULES, Config, apply_rules

ML_WEIGHT = 0.5


def scan(df: pd.DataFrame, cfg: Config | None = None, use_rules: bool = True, use_ml: bool = True) -> pd.DataFrame:
    cfg = cfg or Config()
    hits = apply_rules(df, cfg)
    ml = isolation_scores(df) if use_ml else pd.Series(0.0, index=df.index)

    # Noisy-OR: each independent signal can only add risk, and no score exceeds 1.
    keep = np.ones(len(df))
    if use_rules:
        for r in RULES:
            keep *= np.where(hits[f"rule_{r.name}"], 1 - r.weight, 1.0)
    keep *= 1 - ML_WEIGHT * ml.to_numpy()

    out = pd.concat([df, hits], axis=1)
    out["ml_score"] = ml.round(3)
    out["risk_score"] = (1 - keep).round(3)
    out["reasons"] = _reasons(hits, ml, cfg) if use_rules else ""
    return out.sort_values("risk_score", ascending=False)


def _reasons(hits: pd.DataFrame, ml: pd.Series, cfg: Config) -> pd.Series:
    params = asdict(cfg)
    texts = {r.name: r.reason.format(**params) for r in RULES}
    def row_reasons(i):
        found = [texts[r.name] for r in RULES if hits.at[i, f"rule_{r.name}"]]
        if ml.at[i] > 0.5:
            found.append("statistically unusual pattern (Isolation Forest)")
        return "; ".join(found)
    return pd.Series({i: row_reasons(i) for i in hits.index})
