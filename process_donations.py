#!/usr/bin/env python3
"""Extract donation records from Google Takeout archives.

This script searches for takeout ZIP archives in the current directory,
parses emails from support@sumit.co.il and produces ``donations.csv``.
"""

from __future__ import annotations

import csv
import email
import logging
import mailbox
import re
import shutil
import sys
import tempfile
from datetime import datetime
from email.message import Message
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import zipfile


def collect_zip_parts(root: Path) -> List[Path]:
    """Return paths to full ZIP files.

    Multi-part archives ``*.zip.partN`` are concatenated into a temporary
    file. Single ``*.zip`` files are returned as-is.
    """
    candidates = sorted(root.glob("takeout*.zip*"))
    groups: Dict[str, List[Path]] = {}
    for p in candidates:
        m = re.match(r"(.*\.zip)(?:\.part\d+)?$", p.name)
        if not m:
            continue
        base = m.group(1)
        groups.setdefault(base, []).append(p)

    result: List[Path] = []
    for base, parts in groups.items():
        parts.sort()
        if len(parts) == 1 and parts[0].suffix == ".zip":
            result.append(parts[0])
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with open(tmp.name, "wb") as w:
            for part in parts:
                with open(part, "rb") as r:
                    shutil.copyfileobj(r, w)
        result.append(Path(tmp.name))
    return result


def extract_messages(zip_path: Path) -> Iterable[Message]:
    """Yield email messages from a takeout ZIP."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(temp_dir)
        mboxes = list(temp_dir.rglob("*.mbox"))
        if mboxes:
            for mbox_path in mboxes:
                for msg in mailbox.mbox(mbox_path):
                    yield msg
        else:
            for eml in temp_dir.rglob("*.eml"):
                with open(eml, "rb") as f:
                    yield email.message_from_binary_file(f)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_body(msg: Message) -> str:
    """Return message body as UTF-8 text."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload is not None:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, "ignore")
        return ""
    payload = msg.get_payload(decode=True)
    if payload is None:
        return ""
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, "ignore")


def parse_donation(msg: Message) -> Optional[Dict[str, str]]:
    """Extract donation fields from an email message."""
    sender = msg.get("From", "")
    if not re.search(r"support@sumit\.co\.il", sender, re.I):
        return None
    subject = msg.get("Subject", "")
    if "חיוב" not in subject:
        return None
    body = _get_body(msg)
    name_match = re.search(r"בוצע\s+חיוב\s+עבור\s+([^\s]+)(?:\s|ב-)", body)
    amount_match = re.search(r"([\d,.]+)\s*₪", body)
    date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", body)

    if not (name_match and amount_match and date_match):
        logging.warning("Missing fields in message %s", msg.get("Message-ID"))
        return None

    donor_name = name_match.group(1)
    amount = float(amount_match.group(1).replace(",", ""))
    day, month, year = map(int, date_match.groups())
    donation_date = datetime(year, month, day).strftime("%d-%m-%Y")

    return {
        "donor_name": donor_name,
        "amount": f"{amount:.2f}",
        "currency": "₪",
        "donation_date": donation_date,
        "subject": subject,
        "message_id": msg.get("Message-ID", ""),
    }


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Mark duplicate donations."""
    seen: Dict[tuple, str] = {}
    duplicate_groups = 0
    for row in rows:
        key = (row["donor_name"].lower(), row["amount"], row["donation_date"])
        if key not in seen:
            seen[key] = row["message_id"]
            row["duplicate_of"] = ""
        else:
            if seen[key] == row["message_id"]:
                row["duplicate_of"] = ""
            else:
                row["duplicate_of"] = seen[key]
                duplicate_groups += 1
    logging.info("Duplicate groups: %d", duplicate_groups)
    return rows


def write_csv(rows: List[Dict[str, str]], outfile: Path) -> None:
    """Write rows to CSV."""
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "donor_name",
                "amount",
                "currency",
                "donation_date",
                "subject",
                "message_id",
                "duplicate_of",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path.cwd()
    zip_files = collect_zip_parts(root)
    if not zip_files:
        logging.error("No takeout zip files found")
        sys.exit(1)

    all_rows: List[Dict[str, str]] = []
    processed_msgs = 0

    for z in zip_files:
        logging.info("Processing %s", z)
        for msg in extract_messages(z):
            processed_msgs += 1
            row = parse_donation(msg)
            if row:
                all_rows.append(row)

    if not all_rows:
        logging.error("No matching donation emails found")
        sys.exit(1)

    rows_with_dupes = dedupe_rows(all_rows)
    write_csv(rows_with_dupes, root / "donations.csv")

    unique_keys = {
        (r["donor_name"].lower(), r["amount"], r["donation_date"])
        for r in rows_with_dupes
    }
    logging.info(
        "Processed %d emails, extracted %d donations, %d duplicate groups",
        processed_msgs,
        len(rows_with_dupes),
        len(all_rows) - len(unique_keys),
    )


if __name__ == "__main__":
    main()
