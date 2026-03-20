from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import list_worksheet_titles, reset_workbook_tabs
from production.sheet_contract import MARKETING_TABS, SPREADSHEET_NAME


def main() -> None:
    before = list_worksheet_titles()
    after = reset_workbook_tabs(MARKETING_TABS, rows=3000, cols=60)

    print(f"Spreadsheet: {SPREADSHEET_NAME}")
    print("Before:")
    for title in before:
        print(f"- {title}")
    print("After:")
    for title in after:
        print(f"- {title}")


if __name__ == "__main__":
    main()
