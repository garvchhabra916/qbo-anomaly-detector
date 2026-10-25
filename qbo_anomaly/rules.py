"""Explainable, audit-style rules. Each returns a boolean Series aligned to the input frame.

The weights are a rule's standalone risk: how suspicious one hit is on its own.
They are combined with a noisy-OR in `score.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Config:
    approval_threshold: float = 5000.0
    threshold_margin: float = 0.03  # flag amounts within 3% below the threshold
    duplicate_window_days: int = 7
    round_min_amount: float = 1000.0
    outlier_z: float = 4.0
    min_vendor_history: int = 5
    rare_account_share: float = 0.05
    min_history_for_account_rule: int = 10
    backdate_days: int = 30
    new_vendor_percentile: float = 0.95


def duplicate_payment(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Same vendor and amount within N days of an earlier transaction."""
    prev_date = df.sort_values("txn_date").groupby(["vendor", "amount"])["txn_date"].shift()
    gap = (df["txn_date"] - prev_date.reindex(df.index)).dt.days
    return gap.le(cfg.duplicate_window_days).fillna(False)


def recurring_charge(df: pd.DataFrame) -> pd.Series:
    """Rent, utilities, subscriptions: a repeated exact amount, or a vendor billed about monthly.

    These are usually autopay on a fixed day of the month, so they land on weekends
    and on round numbers without anyone doing anything odd.
    """
    same_amount = df.groupby(["vendor", "amount"])["amount"].transform("size") >= 3
    gaps = df.sort_values("txn_date").groupby("vendor")["txn_date"].diff().dt.days
    median_gap = gaps.groupby(df["vendor"]).transform("median").reindex(df.index)
    return same_amount | median_gap.between(25, 35)


def round_amount(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Large, whole-hundred amounts that aren't a vendor's normal recurring charge."""
    is_round = (df["amount"] >= cfg.round_min_amount) & np.isclose(df["amount"] % 100, 0)
    return is_round & ~recurring_charge(df)


def below_approval_threshold(df: pd.DataFrame, cfg: Config) -> pd.Series:
    lo = cfg.approval_threshold * (1 - cfg.threshold_margin)
    return df["amount"].between(lo, cfg.approval_threshold, inclusive="left")


def weekend_posting(df: pd.DataFrame, cfg: Config) -> pd.Series:
    return (df["txn_date"].dt.day_of_week >= 5) & ~recurring_charge(df)


def vendor_robust_z(df: pd.DataFrame) -> pd.Series:
    """Robust z-score of log(amount) within each vendor (median/MAD, so outliers don't mask themselves)."""
    log_amt = np.log1p(df["amount"])
    grp = log_amt.groupby(df["vendor"])
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    # Floor the MAD so fixed-price vendors (MAD = 0) don't explode on a few cents of difference.
    return 0.6745 * (log_amt - med) / mad.clip(lower=0.05)


def vendor_amount_outlier(df: pd.DataFrame, cfg: Config) -> pd.Series:
    history = df.groupby("vendor")["amount"].transform("size")
    return (vendor_robust_z(df) > cfg.outlier_z) & (history >= cfg.min_vendor_history)


def new_vendor_large_payment(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """A vendor's first-ever payment is in the top tail of all spend."""
    first = ~df.sort_values("txn_date").duplicated("vendor").reindex(df.index)
    big = df["amount"] >= df["amount"].quantile(cfg.new_vendor_percentile)
    history = df.groupby("vendor")["amount"].transform("size")
    return first & big & (history < cfg.min_vendor_history)


def miscategorized_account(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Vendor coded to an account it almost never uses."""
    pair = df.groupby(["vendor", "account"])["amount"].transform("size")
    total = df.groupby("vendor")["amount"].transform("size")
    return (pair / total < cfg.rare_account_share) & (total >= cfg.min_history_for_account_rule)


def backdated_entry(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Entered in QBO long after the transaction date (possible closed-period manipulation)."""
    lag = (df["created_at"] - df["txn_date"]).dt.days
    return lag.gt(cfg.backdate_days).fillna(False)


@dataclass
class Rule:
    name: str
    fn: Callable[[pd.DataFrame, Config], pd.Series]
    weight: float
    reason: str


RULES = [
    Rule("duplicate_payment", duplicate_payment, 0.9, "same vendor & amount within {duplicate_window_days} days"),
    Rule("vendor_amount_outlier", vendor_amount_outlier, 0.8, "amount far above this vendor's norm"),
    Rule("backdated_entry", backdated_entry, 0.7, "entered >{backdate_days} days after transaction date"),
    Rule("below_approval_threshold", below_approval_threshold, 0.6, "just under ${approval_threshold:,.0f} approval limit"),
    Rule("new_vendor_large_payment", new_vendor_large_payment, 0.6, "large first payment to a new vendor"),
    Rule("miscategorized_account", miscategorized_account, 0.6, "vendor rarely coded to this account"),
    Rule("round_amount", round_amount, 0.4, "large round-dollar amount"),
    Rule("weekend_posting", weekend_posting, 0.25, "dated on a weekend"),
]


def apply_rules(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Return one boolean column per rule, named `rule_<name>`."""
    cfg = cfg or Config()
    return pd.DataFrame({f"rule_{r.name}": r.fn(df, cfg).astype(bool) for r in RULES}, index=df.index)


def benford_first_digit(amounts: pd.Series) -> dict:
    """Ledger-level Benford's Law test (Nigrini's MAD conformity thresholds)."""
    a = amounts[amounts >= 10]
    digits = a.astype(str).str.lstrip("0.").str[0].astype(int)
    observed = digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series(np.log10(1 + 1 / np.arange(1, 10)), index=range(1, 10))
    mad = float((observed - expected).abs().mean())
    verdict = ("close conformity" if mad < 0.006 else "acceptable conformity" if mad < 0.012
               else "marginal conformity" if mad < 0.015 else "nonconformity")
    return {"mad": mad, "verdict": verdict, "observed": observed.round(4).to_dict(),
            "expected": expected.round(4).to_dict(), "n": int(len(a))}
