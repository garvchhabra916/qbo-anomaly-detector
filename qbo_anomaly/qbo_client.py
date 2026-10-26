"""Pull Purchase and Bill entities from the QBO Accounting API.

Needs an OAuth 2.0 access token and realm (company) ID. The easiest way to get both
for a sandbox company is Intuit's OAuth 2.0 Playground:
https://developer.intuit.com/app/developer/playground

Usage:
    python -m qbo_anomaly pull --realm 9341452... --out data/live.json
The access token is read from QBO_ACCESS_TOKEN, or asked for without echoing it, so it
never lands in shell history.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SANDBOX = "https://sandbox-quickbooks.api.intuit.com"
PRODUCTION = "https://quickbooks.api.intuit.com"
MINOR_VERSION = "75"
PAGE_SIZE = 1000  # QBO's max per query
ENTITIES = ("Purchase", "Bill")


class QBOError(RuntimeError):
    pass


def _query(base: str, realm: str, token: str, sql: str) -> dict:
    url = f"{base}/v3/company/{realm}/query?" + urllib.parse.urlencode({"query": sql, "minorversion": MINOR_VERSION})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp).get("QueryResponse", {})
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace")[:500]
        if err.code == 401:
            raise QBOError("QBO rejected the access token (401). Tokens expire after 60 minutes; "
                           "get a fresh one from the OAuth 2.0 Playground.") from None
        if err.code == 403:
            raise QBOError("Access denied (403). Check the realm ID matches the company you "
                           "authorized, and that --production is set only for a production company.") from None
        raise QBOError(f"QBO API error {err.code}: {body}") from None


def fetch(entity: str, realm: str, token: str, since: str | None = None, sandbox: bool = True) -> list[dict]:
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


def pull(out: str | Path, realm: str | None = None, since: str | None = None, sandbox: bool = True) -> tuple[Path, dict]:
    realm = realm or os.environ.get("QBO_REALM_ID") or input("Realm ID (company ID): ").strip()
    if not realm.isdigit():
        raise QBOError(f"Realm ID should be all digits, got {realm!r}.")
    if since and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
        raise QBOError("--since must look like YYYY-MM-DD.")
    token = os.environ.get("QBO_ACCESS_TOKEN") or getpass.getpass("Access token (input hidden): ").strip()

    payload = {"QueryResponse": {e: fetch(e, realm, token, since, sandbox) for e in ENTITIES}}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path, {e: len(payload["QueryResponse"][e]) for e in ENTITIES}
