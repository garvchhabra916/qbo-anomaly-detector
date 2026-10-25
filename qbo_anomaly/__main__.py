"""CLI: python -m qbo_anomaly {generate,scan,evaluate,pull}"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import ingest, synth
from .evaluate import MODES, benchmark, evaluate
from .rules import Config, benford_first_digit
from .score import scan

REPORT_COLS = ["txn_id", "txn_date", "vendor", "account", "amount", "risk_score", "reasons"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="qbo_anomaly", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic QBO data + ground truth")
    g.add_argument("--out", default="data")
    g.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("scan", help="score a QBO JSON or CSV export")
    s.add_argument("input")
    s.add_argument("--out", default="flagged.csv")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--approval-limit", type=float, default=5000.0)
    s.add_argument("--top", type=int, default=15)

    e = sub.add_parser("evaluate", help="score synthetic data and compare to ground truth")
    e.add_argument("input")
    e.add_argument("truth")
    e.add_argument("--threshold", type=float, default=0.5)

    b = sub.add_parser("benchmark", help="average metrics over many synthetic companies")
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--threshold", type=float, default=0.5)

    pl = sub.add_parser("pull", help="download Purchases and Bills from the QBO API")
    pl.add_argument("--out", default="data/live.json")
    pl.add_argument("--since", help="YYYY-MM-DD")
    pl.add_argument("--production", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        data, truth = synth.write(args.out, args.seed)
        print(f"wrote {data} and {truth}")

    elif args.cmd == "scan":
        df = ingest.load(args.input)
        scored = scan(df, Config(approval_threshold=args.approval_limit))
        flagged = scored[scored["risk_score"] >= args.threshold]
        flagged[REPORT_COLS].to_csv(args.out, index=False)
        b = benford_first_digit(df["amount"])
        print(f"{len(df)} transactions, {len(flagged)} flagged (risk >= {args.threshold}) -> {args.out}")
        print(f"Benford first-digit MAD {b['mad']:.4f}: {b['verdict']}\n")
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(flagged[REPORT_COLS].head(args.top).to_string(index=False))

    elif args.cmd == "evaluate":
        df = ingest.load(args.input)
        truth = pd.read_csv(args.truth)
        results = {name: evaluate(scan(df, use_rules=r, use_ml=m), truth, args.threshold)
                   for name, r, m in MODES}
        print(json.dumps(results, indent=2))

    elif args.cmd == "benchmark":
        result = benchmark(range(1, args.seeds + 1), args.threshold)
        means = result.xs("mean", axis=1, level=1).T
        stds = result.xs("std", axis=1, level=1).T
        print(f"Mean (± std) over {args.seeds} synthetic companies, threshold {args.threshold}\n")
        print((means.map("{:.3f}".format) + " ± " + stds.map("{:.3f}".format)).to_string())

    elif args.cmd == "pull":
        from .qbo_client import pull
        print(f"wrote {pull(args.out, args.since, sandbox=not args.production)}")


if __name__ == "__main__":
    main()
