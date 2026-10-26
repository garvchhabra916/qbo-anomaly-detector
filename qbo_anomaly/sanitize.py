"""Make text from uploaded files safe to display and to re-export."""
from __future__ import annotations

import re

import pandas as pd


def md(text) -> str:
    """Escape Markdown (and Streamlit's $...$ LaTeX) in text that came from an uploaded file."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", str(text))


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formulas (=, +, -, @) in text cells before offering a CSV download."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "str"]):
        out[col] = out[col].astype(str).str.replace(r"^([=+\-@\t\r])", r"'\1", regex=True)
    return out
