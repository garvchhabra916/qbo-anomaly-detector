"""Load QBO data into one flat DataFrame.

Supported inputs:
  * QBO API JSON: a `QueryResponse` payload (or a bare list) of Purchase / Bill entities.
  * QBO CSV export (Reports > Transaction List by Date), straight from QBO: the report
    title rows above the headers and the total/timestamp rows below are skipped.
    Needs Date and Amount; uses Transaction Type, Num, Name, Split/Account and Create Date
    when present. Only expense-type transactions are kept (see EXPENSE_TYPES).

Output columns: txn_id, txn_type, txn_date, created_at, vendor, account, amount, doc_number.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd

# QBO "Transaction Type" values that are money going out to a vendor. Bill Payments are left
# out on purpose: they settle a Bill already counted, and keeping both would look like a
# duplicate payment. Income, transfers, journal entries and credits are left out too.
EXPENSE_TYPES = {"expense", "check", "bill", "cash expense", "credit card expense", "purchase"}

COLUMNS = ["txn_id", "txn_type", "txn_date", "created_at", "vendor", "account", "amount", "doc_number"]


def _expense_account(entity: dict) -> str | None:
    """Account of the largest expense line (QBO puts the bank/card account at the header)."""
    best, best_amt = None, -1.0
    for line in entity.get("Line", []):
        detail = line.get("AccountBasedExpenseLineDetail") or {}
        name = detail.get("AccountRef", {}).get("name")
        if name and float(line.get("Amount", 0)) > best_amt:
            best, best_amt = name, float(line["Amount"])
    return best


def _entity_to_row(entity: dict, txn_type: str) -> dict:
    vendor_ref = entity.get("EntityRef") or entity.get("VendorRef") or {}
    return {
        "txn_id": f"{txn_type}-{entity['Id']}",
        "txn_type": txn_type,
        "txn_date": entity["TxnDate"],
        "created_at": entity.get("MetaData", {}).get("CreateTime"),
        "vendor": vendor_ref.get("name", "(no vendor)"),
        "account": _expense_account(entity),
        "amount": float(entity["TotalAmt"]),
        "doc_number": entity.get("DocNumber"),
    }


def load_qbo_json(path: str | Path) -> pd.DataFrame:
    return parse_qbo_json(Path(path).read_text(encoding="utf-8-sig"))


def parse_qbo_json(text: str) -> pd.DataFrame:
    payload = json.loads(text)
    if isinstance(payload, dict) and "QueryResponse" in payload:
        payload = payload["QueryResponse"]
    rows = []
    if isinstance(payload, list):
        rows = [_entity_to_row(e, e.get("domain_type", "Purchase")) for e in payload]
    else:
        for txn_type in ("Purchase", "Bill"):
            rows += [_entity_to_row(e, txn_type) for e in payload.get(txn_type, []) if not e.get("Credit")]
    return _normalize(pd.DataFrame(rows, columns=COLUMNS))


def _parse_amount(text: pd.Series) -> pd.Series:
    """'$1,234.56', '-1,234.56' and accounting-style '(1,234.56)' -> float; blanks -> NaN."""
    t = text.astype(str).str.strip()
    negative = t.str.startswith("(") & t.str.endswith(")")
    t = t.str.replace(r"[$,()\s]", "", regex=True)
    return pd.to_numeric(t, errors="coerce").where(~negative, lambda x: -x)


def _find_header(lines: list[str]) -> int:
    """QBO puts the report title, company name and date range above the real header row."""
    for i, row in enumerate(csv.reader(lines[:30])):
        cells = {c.strip().lower() for c in row}
        if "date" in cells and "amount" in cells:
            return i
    raise ValueError("Couldn't find a header row with 'Date' and 'Amount' columns. "
                     "Export Reports > Transaction List by Date from QBO as CSV.")


def load_qbo_csv(path: str | Path) -> pd.DataFrame:
    return parse_qbo_csv(Path(path).read_text(encoding="utf-8-sig"))


def parse_qbo_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header = _find_header(lines)
    raw = pd.read_csv(io.StringIO(text), skiprows=header, dtype=str, skip_blank_lines=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]  # QBO's blank indent column
    total_rows = len(raw)

    # Footer rows (TOTAL, blank lines, "Monday, Oct 21, 2026 03:15 PM GMT-07:00") have no
    # parseable date or no amount.
    dates = pd.to_datetime(raw["Date"], errors="coerce", format="mixed")
    amounts = _parse_amount(raw["Amount"])
    raw = raw[dates.notna() & amounts.notna()]
    dates, amounts = dates[raw.index], amounts[raw.index]
    non_transaction_rows = total_rows - len(raw)

    type_col = next((c for c in ("Transaction Type", "Type") if c in raw), None)
    txn_type = raw[type_col].fillna("").str.strip() if type_col else pd.Series("Expense", index=raw.index)
    expense = txn_type.str.lower().isin(EXPENSE_TYPES)
    skipped_types = txn_type[~expense].value_counts().to_dict()
    raw, txn_type, dates, amounts = raw[expense], txn_type[expense], dates[expense], amounts[expense]

    # In Transaction List, "Account" is the bank/card paid from and "Split" is the expense
    # category ("-Split-" when spread over several categories).
    if "Split" in raw:
        account = raw["Split"].str.strip().replace("-Split-", "(multiple categories)")
    else:
        account = raw.get("Account")
    vendor_col = next((c for c in ("Name", "Payee", "Vendor") if c in raw), None)

    df = pd.DataFrame({
        # Spreadsheet row number, so a flag can be found in the original export.
        "txn_id": [f"row-{header + i + 2}" for i in raw.index],
        "txn_type": txn_type,
        "txn_date": dates,
        "created_at": raw.get("Create Date"),
        "vendor": raw[vendor_col] if vendor_col else None,
        "account": account,
        # Expenses export as negative or positive depending on the report; size is what matters.
        "amount": amounts.abs(),
        "doc_number": raw.get("Num"),
    }, index=raw.index)
    df = _normalize(df)
    df.attrs["skipped"] = {"non_transaction_rows": non_transaction_rows,
                           "non_expense_types": skipped_types}
    return df


def load(path: str | Path) -> pd.DataFrame:
    return load_qbo_csv(path) if str(path).lower().endswith(".csv") else load_qbo_json(path)


def load_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded file in memory, so nothing a visitor uploads is written to disk."""
    text = data.decode("utf-8-sig")
    return parse_qbo_csv(text) if filename.lower().endswith(".csv") else parse_qbo_json(text)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    # QBO API timestamps carry a UTC offset; drop tz so they compare with TxnDate.
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce", format="mixed").dt.tz_localize(None)
    df["vendor"] = df["vendor"].fillna("(no vendor)")
    df["account"] = df["account"].fillna("(uncategorized)")
    return df.sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
