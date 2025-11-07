# utils.py
import pandas as pd
from datetime import datetime
import os

CSV_PATH = "idle_log.csv"

def ensure_csv():
    """Create CSV file with headers if not exists."""
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            "employee_id", "date", "start_time", "end_time", "idle_seconds", "notes"
        ])
        df.to_csv(CSV_PATH, index=False)

def log_idle(employee_id, start_ts, end_ts, notes=""):
    """
    Append an idle event to CSV.
    start_ts and end_ts are epoch seconds (time.time()).
    """
    ensure_csv()
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    idle_seconds = round(end_ts - start_ts, 2)
    row = {
        "employee_id": employee_id,
        "date": start_dt.date().isoformat(),
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "idle_seconds": idle_seconds,
        "notes": notes
    }
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def read_logs():
    ensure_csv()
    return pd.read_csv(CSV_PATH)
# utils.py
import pandas as pd
from datetime import datetime
import os

CSV_PATH = "idle_log.csv"

def ensure_csv():
    """Create CSV file with headers if not exists."""
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            "employee_id", "date", "start_time", "end_time", "idle_seconds", "notes"
        ])
        df.to_csv(CSV_PATH, index=False)

def log_idle(employee_id, start_ts, end_ts, notes=""):
    """
    Append an idle event to CSV.
    start_ts and end_ts are epoch seconds (time.time()).
    """
    ensure_csv()
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    idle_seconds = round(end_ts - start_ts, 2)
    row = {
        "employee_id": employee_id,
        "date": start_dt.date().isoformat(),
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "idle_seconds": idle_seconds,
        "notes": notes
    }
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def read_logs():
    ensure_csv()
    return pd.read_csv(CSV_PATH)
