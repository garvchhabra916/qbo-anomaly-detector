"""Streamlit dashboard: streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from qbo_anomaly import ingest, synth
from qbo_anomaly.evaluate import evaluate
from qbo_anomaly.rules import RULES, Config, benford_first_digit
from qbo_anomaly.sanitize import csv_safe, md
from qbo_anomaly.score import scan

DATA = Path(__file__).parent / "data"
SAMPLE, SAMPLE_TRUTH = DATA / "purchases.json", DATA / "ground_truth.csv"

# Palette: flagged uses the reserved "critical" status color (always paired with a label),
# normal transactions recede in muted gray, single-series charts use categorical slot 1.
# Benford's expected curve is a second series, so it takes categorical slot 2. Value labels use the
# muted ink, which reads on both Streamlit's light and dark themes.
FLAGGED, NORMAL, SERIES, SERIES_2, INK = "#d03b3b", "#898781", "#2a78d6", "#eb6834", "#898781"
RULE_LABELS = {r.name: r.name.replace("_", " ").capitalize() for r in RULES}

st.set_page_config(page_title="QBO Anomaly Detector", page_icon="🔎", layout="wide")


# Uploads are parsed in memory and cached per file content, so one visitor's file is never
# written to disk or shown to anyone else; entries expire after an hour.
@st.cache_data(max_entries=20, ttl=3600, show_spinner="Scanning transactions...")
def load_and_scan(data: bytes, filename: str, threshold_limit: float) -> tuple[pd.DataFrame, dict]:
    df = ingest.load_bytes(data, filename)
    if df.empty:
        raise ValueError("No expense transactions found in this file.")
    return scan(df, Config(approval_threshold=threshold_limit)), df.attrs.get("skipped", {})


@st.cache_data
def scan_one_layer(approval_limit: float, use_rules: bool, use_ml: bool) -> pd.DataFrame:
    return scan(ingest.load(SAMPLE), Config(approval_threshold=approval_limit), use_rules=use_rules, use_ml=use_ml)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("QBO export (API JSON or Transaction List CSV)", type=["json", "csv"])
    if upload is None and not SAMPLE.exists():
        synth.write(DATA)
    st.caption("No file? A synthetic company with 20 planted anomalies is loaded."
               if upload is None else f"Scanning **{md(upload.name)}**")
    st.header("Settings")
    threshold = st.slider("Flag when risk score ≥", 0.1, 0.95, 0.5, 0.05)
    approval_limit = st.number_input("Approval limit ($)", 500.0, 100_000.0, 5000.0, 500.0)

if upload is not None:
    raw, filename, truth = upload.getvalue(), upload.name, None
else:
    raw, filename, truth = SAMPLE.read_bytes(), SAMPLE.name, pd.read_csv(SAMPLE_TRUTH)

try:
    scored, skipped = load_and_scan(raw, filename, approval_limit)
except (ValueError, KeyError) as err:
    st.error(f"Couldn't read **{md(filename)}**: {md(err)}")
    st.stop()
except Exception:
    # Anything unexpected: say so without echoing internals back to the visitor.
    st.error(f"Couldn't read **{md(filename)}**. Export it from QBO as Reports > Transaction List by Date (CSV).")
    st.stop()

if skipped:
    with st.sidebar:
        types = skipped.get("non_expense_types", {})
        if types:
            st.caption("Skipped (not expenses): " + ", ".join(f"{n} {md(t)}" for t, n in sorted(types.items(), key=lambda x: -x[1]))
                       + ". Bill Payments are skipped because the Bill itself is already counted.")
        if skipped.get("non_transaction_rows"):
            st.caption(f"Ignored {skipped['non_transaction_rows']} report title, total and blank rows.")
scored["status"] = (scored["risk_score"] >= threshold).map({True: "Flagged", False: "Normal"})
flagged = scored[scored["status"] == "Flagged"]
rule_cols = [f"rule_{r.name}" for r in RULES]

# ---------- Header + KPIs ----------
st.title("🔎 QBO Anomaly Detector")
st.markdown("Audit rules plus an Isolation Forest, scoring every QuickBooks Online expense "
            "for duplicates, approval-limit gaming, backdating, miscoding and outliers.")

k1, k2, k3, k4 = st.columns(4, border=True)
k1.metric("Transactions scanned", f"{len(scored):,}")
k2.metric("Flagged for review", f"{len(flagged):,}", f"{len(flagged) / len(scored):.1%} of ledger", delta_color="off", delta_arrow="off")
k3.metric("Dollars flagged", f"${flagged['amount'].sum():,.0f}",
          f"of ${scored['amount'].sum():,.0f} total", delta_color="off", delta_arrow="off")
benford = benford_first_digit(scored["amount"])
benford_short = {"close conformity": "Conforms", "acceptable conformity": "Conforms",
                 "marginal conformity": "Marginal", "nonconformity": "Deviates"}[benford["verdict"]]
k4.metric("Benford's Law", benford_short, f"MAD {benford['mad']:.4f}", delta_color="off", delta_arrow="off",
          help="First-digit distribution vs Benford's Law (Nigrini MAD thresholds). See Ledger patterns.")

tab_review, tab_ledger, tab_model = st.tabs(["Review queue", "Ledger patterns", "Model performance"])

# ---------- Review queue ----------
with tab_review:
    st.subheader(f"{len(flagged)} transactions to review")
    st.caption("Sorted by risk. Select a row to see why it was flagged and how it compares with the vendor's history.")
    table = flagged[["risk_score", "txn_date", "vendor", "amount", "account", "reasons", "txn_id"]]
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 420),
        on_select="rerun", selection_mode="single-row",
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f", width="small"),
            "txn_date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "vendor": "Vendor",
            "amount": st.column_config.NumberColumn("Amount", format="dollar", width="small"),
            "account": "Account",
            "reasons": st.column_config.TextColumn("Why it was flagged", width="large"),
            "txn_id": "QBO ID",
        },
    )
    st.download_button("Download review queue (CSV)", csv_safe(table).to_csv(index=False), "flagged.csv", "text/csv")

    rows = event.selection.rows if event and event.selection else []
    if not len(flagged):
        st.info("Nothing above the threshold.")
    else:
        st.divider()
        pick = table.iloc[rows[0]] if rows else table.iloc[0]
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.subheader(f"{md(pick['vendor'])} · \\${pick['amount']:,.2f}")
            st.caption(f"{md(pick['txn_id'])} · {pick['txn_date']:%b %d, %Y} · {md(pick['account'])} · "
                       f"risk {pick['risk_score']:.2f}" + ("" if rows else " · showing the top row"))
            for reason in pick["reasons"].split("; "):
                st.markdown(f"🚩 {md(reason)}")
        with right:
            history = scored[scored["vendor"] == pick["vendor"]].copy()
            history["this"] = history["txn_id"].eq(pick["txn_id"]).map({True: "This transaction", False: "Other payments"})
            st.markdown(f"**All {len(history)} payments to {md(pick['vendor'])}**")
            st.altair_chart(
                alt.Chart(history).mark_circle(size=80, opacity=0.9, stroke="white", strokeWidth=1).encode(
                    x=alt.X("txn_date:T", title=None),
                    y=alt.Y("amount:Q", title=None, axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("this:N", title=None,
                                    scale=alt.Scale(domain=["Other payments", "This transaction"], range=[NORMAL, FLAGGED]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("txn_date:T", title="Date"), alt.Tooltip("amount:Q", title="Amount", format="$,.2f"),
                             alt.Tooltip("account:N", title="Account")],
                ).properties(height=260),
                width="stretch",
            )

# ---------- Ledger patterns ----------
with tab_ledger:
    st.subheader("Every transaction, by date and amount")
    st.altair_chart(
        alt.Chart(scored).mark_circle(stroke="white", strokeWidth=1).encode(
            x=alt.X("txn_date:T", title=None),
            y=alt.Y("amount:Q", title="Amount (log scale)", scale=alt.Scale(type="log"),
                    axis=alt.Axis(format="$,.0f", values=[1, 10, 100, 1000, 10000], grid=True)),
            color=alt.Color("status:N", title=None, scale=alt.Scale(domain=["Normal", "Flagged"], range=[NORMAL, FLAGGED]),
                            legend=alt.Legend(orient="top")),
            size=alt.condition(alt.datum.status == "Flagged", alt.value(110), alt.value(28)),
            opacity=alt.condition(alt.datum.status == "Flagged", alt.value(0.95), alt.value(0.35)),
            order=alt.Order("risk_score:Q"),
            tooltip=[alt.Tooltip("vendor:N", title="Vendor"), alt.Tooltip("txn_date:T", title="Date"),
                     alt.Tooltip("amount:Q", title="Amount", format="$,.2f"), alt.Tooltip("account:N", title="Account"),
                     alt.Tooltip("risk_score:Q", title="Risk", format=".2f"), alt.Tooltip("reasons:N", title="Why")],
        ).properties(height=380).interactive(),
        width="stretch",
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("What triggered the flags")
        counts = (flagged[rule_cols].sum().rename(lambda c: RULE_LABELS[c.removeprefix("rule_")])
                  .rename("count").rename_axis("rule").reset_index())
        counts = counts[counts["count"] > 0]
        base = alt.Chart(counts).encode(
            y=alt.Y("rule:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("count:Q", title="Flagged transactions", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("rule:N", title="Rule"), alt.Tooltip("count:Q", title="Flagged")],
        )
        st.altair_chart(
            (base.mark_bar(color=SERIES, cornerRadiusEnd=4, height=18)
             + base.mark_text(align="left", dx=4, color=INK).encode(text="count:Q")).properties(height=280),
            width="stretch",
        )
        st.caption("A transaction can trip several rules, so bars can sum past the flag count.")

    with c2:
        st.subheader("Benford's Law: first-digit test")
        bf = pd.DataFrame({"digit": range(1, 10),
                           "Observed": [benford["observed"][d] for d in range(1, 10)],
                           "Benford expected": [benford["expected"][d] for d in range(1, 10)]})
        tip = [alt.Tooltip("digit:O", title="First digit"), alt.Tooltip("Observed:Q", format=".1%"),
               alt.Tooltip("Benford expected:Q", format=".1%")]
        bars = alt.Chart(bf).mark_bar(color=SERIES, cornerRadiusEnd=4, size=22).encode(
            x=alt.X("digit:O", title="Leading digit", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Observed:Q", title="Share of transactions", axis=alt.Axis(format="%")), tooltip=tip)
        expected = alt.Chart(bf).mark_line(color=SERIES_2, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=SERIES_2)).encode(
            x="digit:O", y="Benford expected:Q", tooltip=tip)
        st.altair_chart((bars + expected).properties(height=280), width="stretch")
        st.caption(f"Blue bars = this ledger, orange line = Benford's expected curve. Verdict: **{benford['verdict']}** "
                   f"(MAD {benford['mad']:.4f}, Nigrini thresholds). Small books with fixed recurring bills often "
                   "deviate without any fraud, so this is a ledger-level prompt, not a transaction flag.")

# ---------- Model performance ----------
with tab_model:
    if truth is None:
        st.info("Performance needs ground truth, so it's only shown for the built-in synthetic company. "
                "Remove the uploaded file to see it.")
    else:
        st.markdown(f"The sample company has **{len(truth)} planted anomalies** across 8 types. "
                    f"This is how the detector does at the current threshold ({threshold:.2f}). One company is a small "
                    "sample: averaged over 20 synthetic companies, combined F1 is 0.90 "
                    "(`python -m qbo_anomaly benchmark`).")
        modes = ["Rules only", "Isolation Forest only", "Combined"]
        results = {"Rules only": evaluate(scan_one_layer(approval_limit, True, False), truth, threshold),
                   "Isolation Forest only": evaluate(scan_one_layer(approval_limit, False, True), truth, threshold),
                   "Combined": evaluate(scored, truth, threshold)}
        combined = results["Combined"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{combined['precision']:.0%}", help="Of the flags raised, how many were real anomalies")
        m2.metric("Recall", f"{combined['recall']:.0%}", help="Of the planted anomalies, how many were caught")
        m3.metric("F1", f"{combined['f1']:.2f}")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Rules vs ML vs combined")
            comp = pd.DataFrame([{"approach": n, "F1": r["f1"]} for n, r in results.items()])
            base = alt.Chart(comp).encode(
                y=alt.Y("approach:N", sort=modes, title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1]), title="F1 score"),
                tooltip=[alt.Tooltip("approach:N", title="Approach"), alt.Tooltip("F1:Q", format=".2f")])
            st.altair_chart((base.mark_bar(color=SERIES, cornerRadiusEnd=4, height=22)
                             + base.mark_text(align="left", dx=4, color=INK).encode(text=alt.Text("F1:Q", format=".2f")))
                            .properties(height=180), width="stretch")
            st.caption("Generic outlier detection misses context such as *this vendor's* usual amount or the approval "
                       "limit. The domain rules do most of the work, and the model breaks ties.")
        with c2:
            st.subheader("Recall by anomaly type")
            per = pd.DataFrame([{"type": k.replace("_", " ").capitalize(), "recall": v}
                                for k, v in combined["per_type_recall"].items()])
            base = alt.Chart(per).encode(
                y=alt.Y("type:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Caught"),
                tooltip=[alt.Tooltip("type:N", title="Type"), alt.Tooltip("recall:Q", title="Caught", format=".0%")])
            st.altair_chart((base.mark_bar(color=SERIES, cornerRadiusEnd=4, height=18)
                             + base.mark_text(align="left", dx=4, color=INK).encode(text=alt.Text("recall:Q", format=".0%")))
                            .properties(height=280), width="stretch")
            st.caption("Weekend postings are the hardest: real businesses spend on weekends too, "
                       "so that rule only nudges the ranking.")

st.divider()
st.caption("Built with pandas, scikit-learn and Streamlit · data is synthetic unless you upload your own · "
           "nothing leaves this machine")
