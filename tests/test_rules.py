import pandas as pd
import pytest

from qbo_anomaly import rules
from qbo_anomaly.ingest import load_qbo_csv, load_qbo_json
from qbo_anomaly.rules import Config

CFG = Config()


def frame(rows):
    """rows: (date, vendor, account, amount[, created_at])"""
    df = pd.DataFrame([r if len(r) == 5 else (*r, r[0]) for r in rows],
                      columns=["txn_date", "vendor", "account", "amount", "created_at"])
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def test_duplicate_flags_only_the_repeat():
    df = frame([("2026-03-02", "Staples", "Office Supplies", 84.17),
                ("2026-03-05", "Staples", "Office Supplies", 84.17),
                ("2026-04-20", "Staples", "Office Supplies", 84.17)])  # outside the 7-day window
    assert rules.duplicate_payment(df, CFG).tolist() == [False, True, False]


def test_round_amount_ignores_recurring_rent():
    rent = [(f"2026-0{m}-01", "Landlord", "Rent", 6500.0) for m in range(1, 5)]
    df = frame(rent + [("2026-03-10", "Uber", "Travel", 2000.0)])
    assert rules.round_amount(df, CFG).tolist() == [False] * 4 + [True]


def test_below_approval_threshold():
    df = frame([("2026-03-02", "A", "X", 4990.0), ("2026-03-02", "A", "X", 5000.0),
                ("2026-03-02", "A", "X", 4500.0)])
    assert rules.below_approval_threshold(df, CFG).tolist() == [True, False, False]


def test_weekend_ignores_autopay():
    autopay = [(f"2026-0{m}-07", "Intuit", "Software", 90.0) for m in range(1, 6)]  # 2026-03-07 is a Saturday
    df = frame(autopay + [("2026-03-08", "Chipotle", "Meals", 41.20)])  # Sunday
    assert rules.weekend_posting(df, CFG).tolist() == [False] * 5 + [True]


def test_vendor_outlier():
    normal = [(f"2026-01-{d:02d}", "Uber", "Travel", amt) for d, amt in
              zip(range(5, 25, 2), [28.1, 31.4, 35.0, 29.9, 33.3, 30.2, 27.7, 34.8, 32.1, 30.6])]
    df = frame(normal + [("2026-02-02", "Uber", "Travel", 410.0)])
    assert rules.vendor_amount_outlier(df, CFG).tolist() == [False] * 10 + [True]


def test_new_vendor_large_payment():
    base = [(f"2026-01-{d:02d}", "Uber", "Travel", 30.0 + d) for d in range(1, 21)]
    df = frame(base + [("2026-02-01", "Apex Consulting", "Professional Fees", 8000.0)])
    assert rules.new_vendor_large_payment(df, CFG).tolist() == [False] * 20 + [True]


def test_miscategorized_account():
    base = [(f"2026-01-{d:02d}", "Chevron", "Auto & Fuel", 60.0 + d) for d in range(1, 25)]
    df = frame(base + [("2026-02-01", "Chevron", "Meals", 64.0)])
    assert rules.miscategorized_account(df, CFG).tolist() == [False] * 24 + [True]


def test_backdated_entry():
    df = frame([("2026-01-05", "A", "X", 10.0, "2026-01-08"), ("2026-01-05", "A", "X", 10.0, "2026-03-20")])
    assert rules.backdated_entry(df, CFG).tolist() == [False, True]


def test_benford_on_benford_data():
    import numpy as np
    amounts = pd.Series(10 ** np.random.default_rng(0).uniform(1, 5, 20_000))
    assert rules.benford_first_digit(amounts)["verdict"] == "close conformity"


def test_ingest_qbo_json_bill_and_purchase(tmp_path):
    line = {"Amount": 50.0, "DetailType": "AccountBasedExpenseLineDetail",
            "AccountBasedExpenseLineDetail": {"AccountRef": {"value": "7", "name": "Utilities"}}}
    payload = {"QueryResponse": {
        "Purchase": [{"Id": "1", "TxnDate": "2026-01-02", "TotalAmt": 50.0, "EntityRef": {"name": "PG&E"},
                      "Line": [line], "MetaData": {"CreateTime": "2026-01-03T10:00:00-08:00"}}],
        "Bill": [{"Id": "9", "TxnDate": "2026-01-04", "TotalAmt": 50.0, "VendorRef": {"name": "PG&E"},
                  "Line": [line], "MetaData": {"CreateTime": "2026-01-04T10:00:00-08:00"}}],
    }}
    p = tmp_path / "q.json"
    p.write_text(__import__("json").dumps(payload))
    df = load_qbo_json(p)
    assert df["txn_id"].tolist() == ["Purchase-1", "Bill-9"]
    assert set(df["account"]) == {"Utilities"}


def test_ingest_qbo_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text('Date,Transaction Type,Num,Name,Account,Amount\n01/02/2026,Expense,101,Staples,Office Supplies,"-1,204.50"\n')
    df = load_qbo_csv(p)
    assert df.loc[0, "amount"] == pytest.approx(1204.50)
    assert df.loc[0, "vendor"] == "Staples"


FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "qbo_transaction_list.csv"


def test_ingest_raw_qbo_export_skips_title_and_footer_rows():
    df = load_qbo_csv(FIXTURE)
    assert df.attrs["skipped"]["non_transaction_rows"] == 4  # blank, TOTAL, blank, timestamp
    assert df["txn_id"].tolist() == ["row-6", "row-7", "row-9", "row-12"]  # rows in the original file


def test_ingest_keeps_only_expenses():
    df = load_qbo_csv(FIXTURE)
    assert df["txn_type"].tolist() == ["Expense", "Check", "Bill", "Credit Card Expense"]
    assert df.attrs["skipped"]["non_expense_types"] == {
        "Invoice": 1, "Bill Payment (Check)": 1, "Deposit": 1, "Journal Entry": 1}


def test_ingest_uses_split_as_expense_category_and_parses_amounts():
    df = load_qbo_csv(FIXTURE).set_index("vendor")
    assert df.loc["Staples", "account"] == "Office Supplies"  # Split, not the Visa card
    assert df.loc["Amazon", "account"] == "(multiple categories)"
    assert df.loc["Bayview Properties", "amount"] == pytest.approx(6500.00)
    assert df.loc["Amazon", "amount"] == pytest.approx(389.50)  # "($389.50)"


def test_ingest_backdated_entry_from_export():
    df = load_qbo_csv(FIXTURE)
    assert rules.backdated_entry(df, CFG).tolist() == [False, False, False, True]


def test_ingest_rejects_file_without_headers(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Report\nnothing useful here\n")
    with pytest.raises(ValueError, match="header row"):
        load_qbo_csv(p)


def test_qbo_client_pages_through_results(monkeypatch):
    from qbo_anomaly import qbo_client
    calls = []

    def fake_query(base, realm, token, sql):
        calls.append(sql)
        start = int(sql.split("STARTPOSITION ")[1].split()[0])
        n = qbo_client.PAGE_SIZE if start == 1 else 3
        return {"Purchase": [{"Id": str(start + i)} for i in range(n)]}

    monkeypatch.setattr(qbo_client, "_query", fake_query)
    rows = qbo_client.fetch("Purchase", "123", "tok", since="2026-01-01")
    assert len(rows) == qbo_client.PAGE_SIZE + 3
    assert "WHERE TxnDate >= '2026-01-01' STARTPOSITION 1001" in calls[1]


def test_qbo_client_rejects_bad_since():
    from qbo_anomaly import qbo_client
    with pytest.raises(qbo_client.QBOError, match="YYYY-MM-DD"):
        qbo_client.pull("x.json", realm="123", since="2026-01-01' OR 1=1")


def test_md_escapes_markdown_and_latex():
    from qbo_anomaly.sanitize import md
    assert md("[click](http://evil)") == r"\[click\]\(http://evil\)"
    assert md("$5 and $6") == r"\$5 and \$6"
    assert md("<img src=x>") == r"\<img src=x\>"


def test_csv_safe_neutralises_formulas():
    from qbo_anomaly.sanitize import csv_safe
    df = pd.DataFrame({"vendor": ["=HYPERLINK(\"http://evil\")", "+1", "@SUM(A1)", "Staples"], "amount": [1.0, -2.0, 3.0, 4.0]})
    out = csv_safe(df)
    assert out["vendor"].tolist() == ["'=HYPERLINK(\"http://evil\")", "'+1", "'@SUM(A1)", "Staples"]
    assert out["amount"].tolist() == [1.0, -2.0, 3.0, 4.0]  # numbers untouched


def test_load_bytes_parses_in_memory():
    from qbo_anomaly.ingest import load_bytes
    df = load_bytes(FIXTURE.read_bytes(), "export.CSV")
    assert len(df) == 4
