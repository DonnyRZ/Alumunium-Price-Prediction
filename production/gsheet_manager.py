from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from production.sheet_contract import SPREADSHEET_NAME


ROOT = Path(__file__).resolve().parents[1]
LOCAL_KEY_PATH = ROOT / "service_account_key.json"
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def _get_secret_payload():
    env_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if env_json:
        return json.loads(env_json)

    try:
        import streamlit as st  # type: ignore

        if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
            value = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
            if isinstance(value, str):
                return json.loads(value)
            if isinstance(value, dict):
                return dict(value)
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    if LOCAL_KEY_PATH.exists():
        return json.loads(LOCAL_KEY_PATH.read_text())

    raise RuntimeError(
        "Credential Google Sheets tidak ditemukan. "
        "Gunakan GCP_SERVICE_ACCOUNT_JSON atau service_account_key.json lokal."
    )


def get_client():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    creds_payload = _get_secret_payload()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_payload, SCOPES)
    return gspread.authorize(creds)


def get_spreadsheet():
    client = get_client()
    return client.open(SPREADSHEET_NAME)


def get_worksheet(tab_name: str, rows: int = 2000, cols: int = 40):
    spreadsheet = get_spreadsheet()
    try:
        return spreadsheet.worksheet(tab_name)
    except Exception:
        return spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols)


def _normalize_df_for_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]) or pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out.fillna("")


def read_sheet(tab_name: str) -> pd.DataFrame:
    try:
        ws = get_worksheet(tab_name)
    except Exception:
        return pd.DataFrame()

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]
    if not headers:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=headers)


def overwrite_sheet(df: pd.DataFrame, tab_name: str) -> None:
    ws = get_worksheet(tab_name)
    df_out = _normalize_df_for_sheet(df)
    ws.clear()
    if df_out.empty:
        if len(df.columns) > 0:
            ws.update([list(df.columns)])
        return
    ws.update([df_out.columns.tolist()] + df_out.values.tolist())


def append_sheet(df: pd.DataFrame, tab_name: str) -> None:
    if df.empty:
        return
    ws = get_worksheet(tab_name)
    existing = ws.get_all_values()
    df_out = _normalize_df_for_sheet(df)
    if not existing:
        ws.update([df_out.columns.tolist()] + df_out.values.tolist())
        return
    ws.append_rows(df_out.values.tolist())


def upsert_sheet(df_new: pd.DataFrame, tab_name: str, key_columns: Iterable[str]) -> pd.DataFrame:
    existing = read_sheet(tab_name)
    if existing.empty:
        combined = df_new.copy()
    else:
        combined = pd.concat([existing, df_new], ignore_index=True)
    key_columns = list(key_columns)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
    overwrite_sheet(combined, tab_name)
    return combined
