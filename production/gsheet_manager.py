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


def list_worksheet_titles() -> list[str]:
    spreadsheet = get_spreadsheet()
    return [worksheet.title for worksheet in spreadsheet.worksheets()]


def get_worksheet(tab_name: str, rows: int = 2000, cols: int = 40):
    spreadsheet = get_spreadsheet()
    try:
        return spreadsheet.worksheet(tab_name)
    except Exception:
        return spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols)


def delete_worksheet_if_exists(tab_name: str) -> bool:
    spreadsheet = get_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except Exception:
        return False
    spreadsheet.del_worksheet(worksheet)
    return True


def reset_workbook_tabs(required_tabs: Iterable[str], rows: int = 2000, cols: int = 40) -> list[str]:
    spreadsheet = get_spreadsheet()
    required_tabs = list(dict.fromkeys(required_tabs))
    existing = spreadsheet.worksheets()
    existing_titles = [worksheet.title for worksheet in existing]
    created_temp = None

    if len(existing) == 1 and existing[0].title in required_tabs and len(required_tabs) == 1:
        return existing_titles

    # Google Sheets refuses deleting the last remaining worksheet.
    # Create a temporary worksheet whenever the current workbook would
    # otherwise be fully removed during the cleanup pass.
    has_required_sheet_already = any(title in required_tabs for title in existing_titles)
    if len(existing) == 1 or not has_required_sheet_already:
        temp_title = "_tmp_reset_"
        if temp_title in existing_titles:
            created_temp = spreadsheet.worksheet(temp_title)
        else:
            created_temp = spreadsheet.add_worksheet(title=temp_title, rows=max(rows, 10), cols=max(cols, 5))
        existing = spreadsheet.worksheets()

    temp_title = created_temp.title if created_temp is not None else None

    for worksheet in list(existing):
        if worksheet.title == temp_title:
            continue
        if worksheet.title not in required_tabs:
            spreadsheet.del_worksheet(worksheet)

    final_titles = [worksheet.title for worksheet in spreadsheet.worksheets()]
    for tab_name in required_tabs:
        if tab_name not in final_titles:
            spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols)

    if created_temp is not None:
        try:
            worksheet = spreadsheet.worksheet(created_temp.title)
            if worksheet.title not in required_tabs:
                spreadsheet.del_worksheet(worksheet)
        except Exception:
            pass

    return [worksheet.title for worksheet in spreadsheet.worksheets()]


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
