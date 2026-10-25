"""Pull Purchase and Bill entities from the QBO Accounting API.

Needs an OAuth 2.0 access token and realm (company) ID. The easiest way to get both
for a sandbox company is Intuit's OAuth 2.0 Playground:
https://developer.intuit.com/app/developer/playground

Usage:
    QBO_ACCESS_TOKEN=... QBO_REALM_ID=... python -m qbo_anomaly pull --out data/live.json
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

SANDBOX = "https://sandbox-quickbooks.api.intuit.com"
PRODUCTION = "https://quickbooks.api.intuit.com"
MINOR_VERSION = "75"
PAGE_SIZE = 1000  # QBO's max per query


def _query(base: str, realm: str, token: str, sql: str) -> dict:
    url = f"{base}/v3/company/{realm}/query?" + urllib.parse.urlencode({"query": sql, "minorversion": MINOR_VERSION})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["QueryResponse"]


def fetch(entity: str, since: str | None = None, sandbox: bool = True) -> list[dict]:
    token, realm = os.environ["QBO_ACCESS_TOKEN"], os.environ["QBO_REALM_ID"]
    base = SANDBOX if sandbox else PRODUCTION
    where = f" WHERE TxnDate >= '{since}'" if since else ""
    rows, start = [], 1
    while True:
        page = _query(base, realm, token, f"SELECT * FROM {entity}{where} STARTPOSITION {start} MAXRESULTS {PAGE_SIZE}")
        batch = page.get(entity, [])
        rows += batch
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def pull(out: str | Path, since: str | None = None, sandbox: bool = True) -> Path:
    payload = {"QueryResponse": {e: fetch(e, since, sandbox) for e in ("Purchase", "Bill")}}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path
