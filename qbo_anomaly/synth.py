"""Generate a year of realistic QBO Purchase entities with labeled, injected anomalies.

Real client books can't go in a public repo, so this gives the detector a ground truth
to be measured against. Output matches the QBO API `QueryResponse` shape.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Vendor:
    name: str
    accounts: dict[str, float]  # account -> share of this vendor's spend
    median: float
    sigma: float  # lognormal spread
    per_month: float  # average transactions per month
    fixed: bool = False  # recurring exact amount (rent, subscriptions)


VENDORS = [
    Vendor("Bayview Properties", {"Rent or Lease": 1.0}, 6500, 0, 1, fixed=True),
    Vendor("PG&E", {"Utilities": 1.0}, 430, 0.2, 1),
    Vendor("Comcast Business", {"Utilities": 1.0}, 189.99, 0, 1, fixed=True),
    Vendor("Amazon Web Services", {"Software & Subscriptions": 1.0}, 850, 0.25, 1),
    Vendor("Google Workspace", {"Software & Subscriptions": 1.0}, 144, 0, 1, fixed=True),
    Vendor("Intuit", {"Software & Subscriptions": 1.0}, 90, 0, 1, fixed=True),
    Vendor("Staples", {"Office Supplies": 0.8, "Office Equipment": 0.2}, 85, 0.5, 5),
    Vendor("Amazon", {"Office Supplies": 0.7, "Office Equipment": 0.3}, 120, 0.6, 8),
    Vendor("Costco", {"Office Supplies": 0.6, "Meals & Entertainment": 0.4}, 210, 0.4, 2),
    Vendor("Uber", {"Travel": 1.0}, 32, 0.4, 10),
    Vendor("Delta Air Lines", {"Travel": 1.0}, 420, 0.3, 2),
    Vendor("Marriott", {"Travel": 1.0}, 310, 0.3, 2),
    Vendor("Chipotle", {"Meals & Entertainment": 1.0}, 48, 0.35, 6),
    Vendor("Blue Bottle Coffee", {"Meals & Entertainment": 1.0}, 22, 0.3, 8),
    Vendor("FedEx", {"Shipping & Postage": 1.0}, 38, 0.5, 6),
    Vendor("USPS", {"Shipping & Postage": 1.0}, 14, 0.4, 5),
    Vendor("Chevron", {"Auto & Fuel": 1.0}, 64, 0.2, 6),
    Vendor("Smith & Lee CPAs", {"Legal & Professional Fees": 1.0}, 1200, 0.3, 1),
    Vendor("Nolo Legal Group", {"Legal & Professional Fees": 1.0}, 900, 0.4, 0.5),
    Vendor("Facebook Ads", {"Advertising & Marketing": 1.0}, 600, 0.35, 3),
    Vendor("Google Ads", {"Advertising & Marketing": 1.0}, 750, 0.3, 3),
    Vendor("Vistaprint", {"Advertising & Marketing": 1.0}, 140, 0.4, 1),
    Vendor("Hartford Insurance", {"Insurance": 1.0}, 375, 0, 1, fixed=True),
    Vendor("Home Depot", {"Repairs & Maintenance": 0.85, "Office Equipment": 0.15}, 95, 0.6, 2),
    Vendor("Best Buy", {"Office Equipment": 1.0}, 450, 0.5, 0.7),
]

ACCOUNTS = sorted({a for v in VENDORS for a in v.accounts})
APPROVAL_THRESHOLD = 5000.0

ANOMALY_TYPES = [
    "duplicate_payment",
    "round_amount",
    "below_approval_threshold",
    "weekend_posting",
    "vendor_amount_outlier",
    "new_vendor_large_payment",
    "miscategorized_account",
    "backdated_entry",
]


def _business_day(rng: np.random.Generator, start: date, end: date) -> date:
    while True:
        d = start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        if d.weekday() < 5 or rng.random() < 0.03:  # some legitimate weekend activity
            return d


def _entity(txn_id: int, txn_date: date, created: datetime, vendor: str, account: str,
            amount: float, payment_type: str) -> dict:
    amount = round(float(amount), 2)
    return {
        "Id": str(txn_id),
        "domain": "QBO",
        "TxnDate": txn_date.isoformat(),
        "PaymentType": payment_type,
        "TotalAmt": amount,
        "DocNumber": str(10000 + txn_id),
        "AccountRef": {"value": "35", "name": "Business Checking" if payment_type == "Check" else "Company Visa"},
        "EntityRef": {"value": str(zlib.crc32(vendor.encode()) % 10_000), "name": vendor, "type": "Vendor"},
        "Line": [{
            "Id": "1",
            "Amount": amount,
            "DetailType": "AccountBasedExpenseLineDetail",
            "AccountBasedExpenseLineDetail": {"AccountRef": {"value": str(ACCOUNTS.index(account) + 1)
                                                             if account in ACCOUNTS else "99",
                                                             "name": account}},
        }],
        "MetaData": {
            "CreateTime": created.strftime("%Y-%m-%dT%H:%M:%S-07:00"),
            "LastUpdatedTime": created.strftime("%Y-%m-%dT%H:%M:%S-07:00"),
        },
    }


def _created(rng: np.random.Generator, txn_date: date, lag_days: int) -> datetime:
    return datetime.combine(txn_date + timedelta(days=lag_days), time(int(rng.integers(8, 19)), int(rng.integers(0, 60))))


def generate(seed: int = 42, start: date = date(2025, 10, 1), months: int = 12,
             anomaly_rate: float = 0.02) -> tuple[dict, pd.DataFrame]:
    """Return (QBO QueryResponse payload, ground-truth labels DataFrame)."""
    rng = np.random.default_rng(seed)
    end = (pd.Timestamp(start) + pd.DateOffset(months=months) - pd.Timedelta(days=1)).date()
    entities: list[dict] = []
    next_id = 1

    def add(txn_date, vendor, account, amount, lag=None, payment_type=None):
        nonlocal next_id
        lag = int(rng.choice([0, 0, 0, 1, 1, 2, 3, 5, 7, 10])) if lag is None else lag
        payment_type = payment_type or ("Check" if amount > 1000 else "CreditCard")
        entities.append(_entity(next_id, txn_date, _created(rng, txn_date, lag), vendor, account, amount, payment_type))
        next_id += 1
        return entities[-1]

    # Normal activity.
    for v in VENDORS:
        accounts, shares = list(v.accounts), list(v.accounts.values())
        if v.per_month >= 1 and (v.fixed or v.per_month == 1):
            # Recurring monthly bill on a stable day of month.
            dom = int(rng.integers(1, 28))
            for m in range(months):
                d = (pd.Timestamp(start) + pd.DateOffset(months=m)).date().replace(day=dom)
                amt = v.median if v.fixed else v.median * rng.lognormal(0, v.sigma)
                add(d, v.name, accounts[0], amt)
            continue
        for _ in range(rng.poisson(v.per_month * months)):
            add(_business_day(rng, start, end), v.name, rng.choice(accounts, p=shares),
                v.median * rng.lognormal(0, v.sigma))

    n_normal = len(entities)
    n_anomalies = max(len(ANOMALY_TYPES), int(n_normal * anomaly_rate))
    labels = []
    variable = [v for v in VENDORS if not v.fixed]
    new_vendor_names = iter(["Apex Consulting LLC", "J. Rivera Services", "Northstar Solutions",
                             "Quick Fix Contractors", "Summit Advisory", "Pinnacle Supply Co",
                             "Redline Partners", "Westgate Holdings"])

    for i in range(n_anomalies):
        kind = ANOMALY_TYPES[i % len(ANOMALY_TYPES)]
        v = variable[int(rng.integers(len(variable)))]
        d = _business_day(rng, start, end)
        typical = v.median * rng.lognormal(0, v.sigma)
        primary = max(v.accounts, key=v.accounts.get)

        if kind == "duplicate_payment":
            src = entities[int(rng.integers(n_normal))]
            src_date = date.fromisoformat(src["TxnDate"])
            e = add(min(src_date + timedelta(days=int(rng.integers(1, 6))), end), src["EntityRef"]["name"],
                    src["Line"][0]["AccountBasedExpenseLineDetail"]["AccountRef"]["name"], src["TotalAmt"])
        elif kind == "round_amount":
            e = add(d, v.name, primary, float(rng.choice([1500, 2000, 2500, 3000, 4000, 7500, 10000])))
        elif kind == "below_approval_threshold":
            e = add(d, v.name, primary, APPROVAL_THRESHOLD - rng.uniform(1, 120))
        elif kind == "weekend_posting":
            d = d + timedelta(days=(5 - d.weekday()) % 7 + int(rng.integers(0, 2)))
            e = add(min(d, end), v.name, primary, typical)
        elif kind == "vendor_amount_outlier":
            e = add(d, v.name, primary, typical * rng.uniform(6, 15))
        elif kind == "new_vendor_large_payment":
            e = add(d, next(new_vendor_names, f"Vendor {i}"), "Legal & Professional Fees", rng.uniform(2800, 9500))
        elif kind == "miscategorized_account":
            # Pick a vendor with enough history for "unusual" to mean something.
            v = max(variable, key=lambda x: x.per_month * rng.random())
            e = add(d, v.name, rng.choice([a for a in ACCOUNTS if a not in v.accounts]), v.median * rng.lognormal(0, v.sigma))
        elif kind == "backdated_entry":
            e = add(d, v.name, primary, typical, lag=int(rng.integers(45, 150)))
        labels.append({"txn_id": f"Purchase-{e['Id']}", "anomaly_type": kind})

    # Shuffle Ids so anomalies aren't identifiable by Id order.
    order = rng.permutation(len(entities))
    remap = {}
    for new_id, idx in enumerate(order, start=1):
        remap[f"Purchase-{entities[idx]['Id']}"] = f"Purchase-{new_id}"
        entities[idx]["Id"] = str(new_id)
        entities[idx]["DocNumber"] = str(10000 + new_id)
    entities.sort(key=lambda e: (e["TxnDate"], int(e["Id"])))
    truth = pd.DataFrame(labels)
    truth["txn_id"] = truth["txn_id"].map(remap)

    payload = {"QueryResponse": {"Purchase": entities, "startPosition": 1, "maxResults": len(entities)},
               "time": datetime.now().isoformat()}
    return payload, truth


def to_transaction_list_csv(payload: dict, path: str | Path, company: str = "Sample Bookkeeping Co") -> Path:
    """Write the ledger the way QBO exports Reports > Transaction List by Date, title rows,
    income rows and footer included, so the CSV upload path can be tried end to end."""
    import csv

    def us_date(iso: str) -> str:
        return datetime.fromisoformat(iso[:10]).strftime("%m/%d/%Y")

    rows = []
    for e in payload["QueryResponse"]["Purchase"]:
        paid_by_check = e["PaymentType"] == "Check"
        rows.append([e["TxnDate"], "Check" if paid_by_check else "Expense", e["DocNumber"] if paid_by_check else "",
                     e["EntityRef"]["name"], e["AccountRef"]["name"],
                     e["Line"][0]["AccountBasedExpenseLineDetail"]["AccountRef"]["name"], -e["TotalAmt"],
                     datetime.fromisoformat(e["MetaData"]["CreateTime"][:19]).strftime("%m/%d/%Y %I:%M:%S %p")])
    # Income side, which the loader should skip.
    for i, month in enumerate(pd.date_range(rows[0][0], periods=12, freq="MS")):
        d = month.date().isoformat()
        rows.append([d, "Invoice", str(1001 + i), "Acme Corp", "Accounts Receivable (A/R)", "Services", 4800.0, ""])
        rows.append([d, "Deposit", "", "", "Business Checking", "Undeposited Funds", 4800.0, ""])
    rows.sort(key=lambda r: r[0])

    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows([["Transaction List by Date"], [company],
                     [f"{us_date(rows[0][0])} - {us_date(rows[-1][0])}"], []])
        w.writerow(["", "Date", "Transaction Type", "Num", "Name", "Account", "Split", "Amount", "Create Date"])
        for r in rows:
            w.writerow(["", us_date(r[0]), *r[1:6], f"{r[6]:,.2f}", r[7]])
        w.writerows([[], ["TOTAL", "", "", "", "", "", "", f"{sum(r[6] for r in rows):,.2f}", ""], [],
                     [datetime.now().strftime("%A, %b %d, %Y %I:%M %p GMT-07:00")]])
    return path


def write(out_dir: str | Path, seed: int = 42) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload, truth = generate(seed=seed)
    data_path, truth_path = out / "purchases.json", out / "ground_truth.csv"
    data_path.write_text(json.dumps(payload, indent=1))
    truth.to_csv(truth_path, index=False)
    to_transaction_list_csv(payload, out / "sample_qbo_export.csv")
    return data_path, truth_path
