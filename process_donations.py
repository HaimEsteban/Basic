#!/usr/bin/env python3
"""Process donation emails from zipped repositories.

This script clones or downloads a GitHub repository, extracts zip files
within it, parses Gmail MBOX files for donation records, cleans the data,
and produces analytical outputs.
"""

from __future__ import annotations

import argparse
import io
import mailbox
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
import matplotlib

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JERUSALEM_TZ = "Asia/Jerusalem"
MEMORIAL_DATES = {
    2022: datetime(2022, 5, 3, tzinfo=pd.Timestamp(0, tz=JERUSALEM_TZ).tz),
    2023: datetime(2023, 4, 25, tzinfo=pd.Timestamp(0, tz=JERUSALEM_TZ).tz),
    2024: datetime(2024, 5, 12, tzinfo=pd.Timestamp(0, tz=JERUSALEM_TZ).tz),
    2025: datetime(2025, 5, 1, tzinfo=pd.Timestamp(0, tz=JERUSALEM_TZ).tz),
}

@dataclass
class Donation:
    donor_raw: str
    donor_id: str
    timestamp: pd.Timestamp
    amount_raw: str
    currency: str
    amount: float
    is_refund: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Donation email processor")
    p.add_argument("--repo-url", required=True, help="GitHub repository URL")
    p.add_argument("--branch", default="main", help="Branch or tag to fetch")
    p.add_argument(
        "--memorial-table-only",
        action="store_true",
        help="Skip charts for quick runs",
    )
    return p.parse_args()


def clone_or_download(repo_url: str, branch: str) -> Path:
    """Clone repo depth=1 or download zip if git unavailable."""
    tmpdir = Path(tempfile.mkdtemp())
    zip_path = tmpdir / "repo.zip"
    try:
        from subprocess import run

        res = run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                str(tmpdir / "repo"),
            ],
            check=True,
            capture_output=True,
        )
        repo_dir = tmpdir / "repo"
    except Exception:
        # Fall back to downloading archive
        archive_url = repo_url.rstrip("/") + f"/archive/{branch}.zip"
        resp = requests.get(archive_url, timeout=30)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        repo_dir = next(tmpdir.glob("*/"))
    return repo_dir


def find_zip_files(path: Path) -> List[Path]:
    """Recursively find zip files."""
    return [p for p in path.rglob("*.zip") if p.is_file()]


def extract_mboxes_from_zip(zip_file: Path, temp_root: Path) -> List[Path]:
    """Unzip and locate mbox files."""
    out_dir = temp_root / zip_file.stem
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(out_dir)
    return [p for p in out_dir.rglob("*.mbox")]


def normalise_donor(name: str) -> str:
    """Return canonical donor identifier."""
    cleaned = re.sub(r"[\s\p{P}]+", " ", name, flags=re.UNICODE)
    return cleaned.strip().casefold()


def parse_amount(text: str) -> tuple[float, str]:
    """Extract numeric amount and currency."""
    m = re.search(r"([\d.,]+)\s*([₪€$])", text)
    if not m:
        raise ValueError("no amount")
    num, curr = m.groups()
    num = float(num.replace(",", ""))
    return num, curr


def parse_mbox(mbox_path: Path) -> Iterable[Donation]:
    """Parse donations from an mbox."""
    mbox = mailbox.mbox(mbox_path)
    for msg in mbox:
        try:
            sender = msg.get("From", "")
            subj = msg.get("Subject", "")
            payload = msg.get_payload(decode=True)
            body = ""
            if isinstance(payload, bytes):
                body = payload.decode("utf-8", "ignore")
            trigger = (
                sender == "support@sumit.co.il"
                or "בוצע חיוב" in subj
                or "זיכוי" in subj
                or "בוצע חיוב" in body
                or "זיכוי" in body
            )
            if not trigger:
                continue
            text = subj + "\n" + body
            donor_match = re.search(r"עבור\s+(.+?)\s*(?:\n|$| )", text)
            if not donor_match:
                continue
            donor_raw = donor_match.group(1).strip()
            amount_raw_match = re.search(r"([\d.,]+\s*[₪€$])", text)
            if not amount_raw_match:
                continue
            amount_raw = amount_raw_match.group(1)
            amount, curr = parse_amount(amount_raw)
            date_header = msg.get("Date")
            if not date_header:
                continue
            ts = pd.to_datetime(date_header, utc=True).tz_convert(JERUSALEM_TZ)
            is_refund = "זיכוי" in text
            yield Donation(
                donor_raw=donor_raw,
                donor_id=normalise_donor(donor_raw),
                timestamp=ts,
                amount_raw=amount_raw,
                currency=curr,
                amount=amount,
                is_refund=is_refund,
            )
        except Exception:
            continue


def load_ecb_rates(date: pd.Timestamp) -> dict:
    """Fetch currency rates from exchangerate.host."""
    url = "https://api.exchangerate.host/" + date.strftime("%Y-%m-%d")
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        # fallback to latest
        resp = requests.get("https://api.exchangerate.host/latest", timeout=10)
    data = resp.json()
    return data.get("rates", {})


def convert_to_ils(amount: float, currency: str, date: pd.Timestamp, rate_cache: dict) -> float:
    """Convert amount to ILS using cached rates."""
    if currency == "₪":
        return amount
    day = date.normalize()
    if day not in rate_cache:
        rate_cache[day] = load_ecb_rates(day)
    rates = rate_cache[day]
    if currency == "€":
        eur_rate = rates.get("ILS")
        return amount * eur_rate if eur_rate else np.nan
    if currency == "$":
        usd_rate = rates.get("ILS") / rates.get("USD") if rates.get("USD") else None
        return amount * usd_rate if usd_rate else np.nan
    return np.nan


def process_repo(repo_dir: Path) -> pd.DataFrame:
    """Main processing: find zips, parse mboxes, clean data."""
    zips = find_zip_files(repo_dir)
    if len(zips) != 2:
        print("Error: repo must contain exactly two zip files", file=sys.stderr)
        sys.exit(1)
    temp_root = Path(tempfile.mkdtemp())
    donations: List[Donation] = []
    for z in zips:
        for mbox_path in extract_mboxes_from_zip(z, temp_root):
            donations.extend(list(parse_mbox(mbox_path)))
    if not donations:
        return pd.DataFrame()
    df = pd.DataFrame([d.__dict__ for d in donations])
    # currency conversion
    rate_cache: dict = {}
    df["amount_ILS"] = df.apply(
        lambda r: convert_to_ils(r["amount"], r["currency"], r["timestamp"], rate_cache),
        axis=1,
    )
    df.dropna(subset=["amount_ILS", "timestamp"], inplace=True)
    df.loc[df["is_refund"], "amount_ILS"] *= -1
    df["year"] = df["timestamp"].dt.year
    df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["days_from_memorial"] = df.apply(
        lambda r: (r["timestamp"].normalize() - MEMORIAL_DATES.get(r["year"], pd.NaT)).days,
        axis=1,
    )
    return df


def aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate statistics."""
    data = {
        "total_rows": [len(df)],
        "unique_donors": [df["donor_id"].nunique()],
        "sum_ils": [df["amount_ILS"].sum()],
        "mean": [df["amount_ILS"].mean()],
        "median": [df["amount_ILS"].median()],
    }
    return pd.DataFrame(data)


def quarterly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly aggregation 2022Q1-2025Q4."""
    df_q = (
        df.groupby(["year", "quarter"])
        .agg(donations=("donor_id", "count"), donors=("donor_id", "nunique"), sum_ils=("amount_ILS", "sum"))
        .reset_index()
    )
    idx = pd.period_range("2022Q1", "2025Q4", freq="Q")
    df_q["period"] = pd.PeriodIndex(df_q["quarter"])
    df_q = df_q.set_index("period").reindex(idx, fill_value=0).reset_index()
    df_q["quarter"] = df_q["period"].astype(str)
    df_q["year"] = df_q["period"].year
    df_q.drop(columns="period", inplace=True)
    return df_q


def peak_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Detect peaks in daily donation sums."""
    daily = df.set_index("timestamp").resample("D").agg(sum_ils=("amount_ILS", "sum"), donors=("donor_id", "nunique"))
    peaks, _ = find_peaks(daily["sum_ils"], distance=7)
    top_peaks = daily.iloc[peaks].nlargest(5, "sum_ils").reset_index()
    top_peaks.rename(columns={"timestamp": "date"}, inplace=True)
    return top_peaks


def memorial_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stats around Yom HaZikaron."""
    records = []
    for year, mem_date in MEMORIAL_DATES.items():
        year_df = df[df["year"] == year]
        win = year_df[(year_df["days_from_memorial"] >= -14) & (year_df["days_from_memorial"] <= 14)]
        base = year_df[(year_df["days_from_memorial"] >= -42) & (year_df["days_from_memorial"] <= -28)]
        win_sum = win["amount_ILS"].sum()
        base_sum = base["amount_ILS"].sum()
        pct_change = ((win_sum - base_sum) / base_sum * 100) if base_sum else np.nan
        records.append({
            "year": year,
            "window_sum": win_sum,
            "window_count": len(win),
            "base_sum": base_sum,
            "pct_change": pct_change,
        })
    return pd.DataFrame(records)


def generate_narrative(quarterly: pd.DataFrame, peaks: pd.DataFrame, mem_stats: pd.DataFrame) -> str:
    """Generate ~300 word Hebrew narrative about donation trends."""
    lines = []
    lines.append("בשנים האחרונות חלו תנודות מעניינות בתרומות." )
    if not quarterly.empty:
        max_q = quarterly.iloc[quarterly['sum_ils'].idxmax()]
        lines.append(f"השיא נרשם ברבעון {max_q['quarter']} עם סכום של כ{int(max_q['sum_ils'])} ש""ח.")
    if not peaks.empty:
        peak = peaks.iloc[0]
        lines.append(f"היום הבולט ביותר הוא {peak['date'].date()} בו נתרמו {int(peak['sum_ils'])} ש""ח.")
    for _, row in mem_stats.iterrows():
        if not np.isnan(row['pct_change']):
            lines.append(
                f"בסביבת יום הזיכרון {row['year']} חלה עלייה של {row['pct_change']:.1f}% ביחס לתקופה המקבילה." )
    return " ".join(lines)


def create_report(df: pd.DataFrame, aggregates_df: pd.DataFrame, quarterly_df: pd.DataFrame,
                   peaks_df: pd.DataFrame, mem_df: pd.DataFrame, narrative: str,
                   memorial_only: bool, outfile: Path) -> None:
    """Generate markdown or notebook report."""
    if outfile.suffix == ".ipynb":
        from nbformat import v4 as nbf

        nb = nbf.new_notebook()
        cells = []
        cells.append(nbf.new_markdown_cell("## דו""ח תרומות"))
        cells.append(nbf.new_markdown_cell(aggregates_df.to_markdown()))
        cells.append(nbf.new_markdown_cell(quarterly_df.to_markdown()))
        if not memorial_only:
            fig, ax = plt.subplots()
            quarterly_df.plot.bar(x="quarter", y="sum_ils", ax=ax)
            fig.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            plt.close(fig)
            img_buf.seek(0)
            cells.append(nbf.new_markdown_cell("![](attachment:quarterly.png)"))
            nb["attachments"] = {
                "quarterly.png": {
                    "image/png": img_buf.getvalue()
                }
            }
        cells.append(nbf.new_markdown_cell(mem_df.to_markdown()))
        cells.append(nbf.new_markdown_cell(peaks_df.to_markdown()))
        cells.append(nbf.new_markdown_cell(narrative))
        nb["cells"] = cells
        outfile.write_text(nbf.writes(nb))
    else:
        with outfile.open("w", encoding="utf-8") as f:
            f.write("## דו""ח תרומות\n")
            f.write(aggregates_df.to_markdown() + "\n")
            f.write(quarterly_df.to_markdown() + "\n")
            if not memorial_only:
                fig, ax = plt.subplots()
                quarterly_df.plot.bar(x="quarter", y="sum_ils", ax=ax)
                fig.tight_layout()
                img_path = outfile.with_name("quarterly.png")
                fig.savefig(img_path)
                plt.close(fig)
                f.write(f"![]({img_path.name})\n")
            f.write(mem_df.to_markdown() + "\n")
            f.write(peaks_df.to_markdown() + "\n")
            f.write(narrative)


def main() -> None:
    args = parse_args()
    repo_dir = clone_or_download(args.repo_url, args.branch)
    df = process_repo(repo_dir)
    if df.empty:
        print("No donations found")
        sys.exit(1)
    df.sort_values("timestamp", inplace=True)
    df.to_csv("donations_clean.csv", index=False)
    agg_df = aggregates(df)
    q_df = quarterly_table(df)
    peaks_df = peak_detection(df)
    mem_df = memorial_stats(df)
    narrative = generate_narrative(q_df, peaks_df, mem_df)
    report_name = "donations_report.ipynb"
    try:
        import nbformat

        ext = ".ipynb"
    except Exception:
        ext = ".md"
        report_name = "donations_report.md"
    create_report(df, agg_df, q_df, peaks_df, mem_df, narrative, args.memorial_table_only, Path(report_name))
    print(f"✨ DONE.  CSV -> donations_clean.csv, Report -> {report_name}")


if __name__ == "__main__":
    main()
