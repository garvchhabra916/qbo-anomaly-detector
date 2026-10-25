"""Unsupervised layer: Isolation Forest over engineered bookkeeping features.

Catches odd combinations no single rule encodes (e.g. a normal-sized amount, at an
unusual vendor/account pairing, entered late, on a weekend).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .rules import recurring_charge, vendor_robust_z


def features(df: pd.DataFrame) -> pd.DataFrame:
    vendor_n = df.groupby("vendor")["amount"].transform("size")
    pair_n = df.groupby(["vendor", "account"])["amount"].transform("size")
    lag = (df["created_at"] - df["txn_date"]).dt.days.fillna(0)
    recurring = recurring_charge(df)
    return pd.DataFrame({
        "log_amount": np.log1p(df["amount"]),
        "vendor_z": vendor_robust_z(df).clip(-10, 10),
        "log_vendor_history": np.log1p(vendor_n),
        "account_share_for_vendor": pair_n / vendor_n,
        "entry_lag_days": lag.clip(-30, 365),
        "is_recurring": recurring.astype(int),
        # Whole-dollar amounts are only odd for non-recurring spend. Weekend is deliberately
        # not a feature: the weekend rule already scores it, and counting it twice tripled
        # false positives in the multi-seed benchmark (precision 0.95 -> 0.67).
        "cents_zero": (np.isclose(df["amount"] % 1, 0) & ~recurring).astype(int),
        "day_of_month": df["txn_date"].dt.day,
    }, index=df.index)


def isolation_scores(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """0-1 score; 0 for the bulk of the data, rising to 1 for the most isolated ~10%."""
    X = features(df)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(X)
    raw = -model.score_samples(X)  # higher = more anomalous
    pct = pd.Series(raw, index=df.index).rank(pct=True)
    return ((pct - 0.9) / 0.1).clip(0, 1)
