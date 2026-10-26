# QBO Anomaly Detector

[![tests](https://github.com/garvchhabra916/qbo-anomaly-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/garvchhabra916/qbo-anomaly-detector/actions/workflows/tests.yml)

**[▶ Live demo](https://qbo-anomaly-detector-6yyiwzui24bth9ncyqp7i4.streamlit.app/)**: try the dashboard in your browser, no install needed.

Flags suspicious expense transactions in QuickBooks Online: duplicate payments, amounts just under an approval limit, backdated entries, miscoded accounts, and more. Every flag comes with a plain-English reason a bookkeeper can act on.

It combines **audit-style rules**, which encode what a bookkeeper already looks for, with an **Isolation Forest**, which catches odd combinations no single rule covers. The two are merged into one ranked 0–1 risk score.

## Results

Benchmarked on 20 synthetic small-business ledgers (~1,000 transactions/year each, 2% injected anomalies across 8 fraud/error types), threshold 0.5:

| Mode | Precision | Recall | F1 |
|---|---|---|---|
| Rules only | 0.951 ± 0.057 | 0.849 ± 0.052 | 0.896 |
| Isolation Forest only | 1.000 ± 0.000 | 0.055 ± 0.002 | 0.104 |
| **Combined** | **0.943 ± 0.058** | **0.863 ± 0.051** | **0.900** |

Reproduce with `python -m qbo_anomaly benchmark`.

**What the numbers say**
- Unsupervised ML alone is close to useless here. Bookkeeping anomalies depend on context (*this vendor*, *this account*, *this approval limit*), and generic outlier detection doesn't see that context. Domain rules do most of the work.
- The ML layer works as a tiebreaker. It lifts borderline rule hits, such as a round amount at a vendor with short history, over the threshold.
- Weekend postings are basically undetectable alone, because real businesses do some weekend spending. The weekend rule only adds to the ranking. An early version also fed weekend into the model, which counted it twice and dropped precision from 0.95 to 0.67. The benchmark caught that.
- Recurring autopay (rent, subscriptions, utilities) is detected and exempted from the weekend and round-number rules. Without that exemption, rent alone produced most of the false positives.

## Detections

| Rule | Weight | What it catches |
|---|---|---|
| `duplicate_payment` | 0.9 | Same vendor + amount within 7 days |
| `vendor_amount_outlier` | 0.8 | Robust z-score (median/MAD on log amount) > 4 within the vendor |
| `backdated_entry` | 0.7 | `MetaData.CreateTime` > 30 days after `TxnDate` (closed-period edits) |
| `below_approval_threshold` | 0.6 | Within 3% under the approval limit (`--approval-limit`) |
| `new_vendor_large_payment` | 0.6 | First payment to a vendor is in the top 5% of all spend |
| `miscategorized_account` | 0.6 | Vendor coded to an account it uses < 5% of the time |
| `round_amount` | 0.4 | ≥ $1,000 whole-hundred amount, not a recurring charge |
| `weekend_posting` | 0.25 | Weekend date, not a recurring charge |
| Isolation Forest | up to 0.5 | Top ~10% most isolated on 8 engineered features |

Scores combine as a noisy-OR: `risk = 1 − ∏(1 − wᵢ)`. Each extra signal adds risk, and the total stays below 1.

`scan` also runs a ledger-wide **Benford's Law** first-digit test, using Nigrini's MAD thresholds. On the synthetic data it reports *nonconformity*. That's expected: a ledger with ~25 vendors and fixed recurring bills doesn't produce Benford-distributed amounts. It's a real caveat for small books.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m qbo_anomaly generate                     # data/purchases.json, ground_truth.csv, sample_qbo_export.csv
python -m qbo_anomaly scan data/purchases.json     # ranked flags -> flagged.csv
python -m qbo_anomaly evaluate data/purchases.json data/ground_truth.csv
python -m qbo_anomaly benchmark --seeds 20
pytest
```

**Dashboard**

```bash
streamlit run app.py
```

The dashboard opens on the synthetic sample company. You can also upload a QBO JSON or CSV export. To try the CSV path, upload `data/sample_qbo_export.csv`, the same company in QBO's export format.
- **Review queue:** flagged transactions ranked by risk. Select one to see each reason and a chart of that vendor's full payment history.
- **Ledger patterns:** every transaction plotted by date and amount, which rules fired, and a Benford's Law chart.
- **Model performance:** precision, recall and F1 at the current threshold, rules vs ML vs combined, and recall by anomaly type.

The sidebar slider changes the flag threshold and the approval limit live.

To share it publicly, push the repo to GitHub and deploy it free on [Streamlit Community Cloud](https://streamlit.io/cloud), pointing it at `app.py`.

**Real QBO data**
- **CSV export:** in QBO, go to Reports → *Transaction List by Date*, choose a date range (a full year works best), and export to CSV. Upload the file as is, or run `scan export.csv`.
  - The report title rows, the TOTAL row and the timestamp footer are skipped automatically.
  - Only expense types are scored: Expense, Check, Bill, Credit Card Expense, Cash Expense. Invoices, deposits, payments, transfers, journal entries and credits are skipped, and the dashboard sidebar shows how many of each. Bill Payments are skipped too, because the Bill they pay is already counted; keeping both would look like a duplicate payment.
  - The expense category comes from the `Split` column. In this report `Account` is the bank or card the money came from.
  - To enable the backdating rule, add the `Create Date` column with Customize → Rows/Columns before exporting.
- **API:** create a free Intuit developer account, then get a sandbox token and realm ID from the [OAuth 2.0 Playground](https://developer.intuit.com/app/developer/playground). Then:
  ```bash
  QBO_ACCESS_TOKEN=... QBO_REALM_ID=... python -m qbo_anomaly pull --out data/live.json
  python -m qbo_anomaly scan data/live.json
  ```
  `pull` pages through `Purchase` and `Bill` via the v3 query endpoint.

## Layout

```
qbo_anomaly/
  ingest.py      QBO API JSON (Purchase, Bill) and raw Transaction List CSV -> one flat DataFrame of expenses
  synth.py       Synthetic QBO-shaped ledgers with labeled injected anomalies
  rules.py       Audit rules + Benford test
  ml.py          Feature engineering + Isolation Forest
  score.py       Noisy-OR combination + human-readable reasons
  evaluate.py    Precision/recall/F1, per-type recall, multi-seed benchmark
  qbo_client.py  QBO Accounting API client (OAuth bearer token)
tests/           Unit tests for every rule and both ingest paths, incl. a raw QBO export fixture
app.py           Streamlit dashboard
```

## Security

The public dashboard is **read-only and stateless**.
- **No database:** the app keeps no records, so a visitor has nothing to change.
- **No credentials:** it holds no QBO tokens or API keys. `pull` runs only on your own machine, and the token is typed in with input hidden, never stored.
- **Uploads are private:** files are parsed in memory and never written to disk. Each upload is scored only for the visitor who sent it, and cached results expire after an hour.
- **Upload limit:** 10 MB.
- **Escaped output:** text from uploaded files (vendor names, accounts) is escaped before display, so it can't inject links, HTML or LaTeX. CSV downloads neutralise spreadsheet formulas (`=`, `+`, `-`, `@`).
- **No tracebacks:** error details are hidden from visitors (`showErrorDetails = "none"`), and Streamlit's XSRF protection stays on.
- **Dependency checks:** CI runs `pip-audit` against known CVEs on every push, and Dependabot opens weekly update PRs.
- **No secrets in the repo:** real client books (`data/live.json`, `data/real/`) are git-ignored.

## Limitations / next steps
- Results are on synthetic data. Running it on a real sandbox company, or anonymized client books, is the real test.
- Rule weights are hand-set. With a few hundred reviewer-labeled flags, they could be learned with logistic regression.
- Only expense-side entities for now. `JournalEntry` and `Deposit` would cover manual adjusting entries and revenue-side fraud.
