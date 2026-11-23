import argparse
import csv
import hashlib
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "legacy_candles.csv"
DEFAULT_DB = PROJECT_DIR / "data" / "iq_ai_data.sqlite3"
EXPECTED_PAIR = "EURUSD"
PRICE_COLUMNS = ("open", "high", "low", "close")
PRICE_TOLERANCE = 1e-9


def load_candles(csv_path):
    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    rows = []
    seen_timestamps = set()

    with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"pair", "ts", *PRICE_COLUMNS, "volume"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain these columns: {', '.join(sorted(required))}")

        for line_number, row in enumerate(reader, start=2):
            pair = (row["pair"] or "").strip().upper()
            if pair != EXPECTED_PAIR:
                raise ValueError(f"Unexpected pair on CSV line {line_number}: {pair!r}")

            try:
                ts = int(row["ts"])
                prices = {name: float(row[name]) for name in PRICE_COLUMNS}
                volume = float(row["volume"] or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc

            if ts <= 0 or ts % 60:
                raise ValueError(f"Timestamp is not a positive minute boundary on line {line_number}")
            if ts in seen_timestamps:
                raise ValueError(f"Duplicate timestamp on CSV line {line_number}: {ts}")
            seen_timestamps.add(ts)

            if not all(math.isfinite(value) and value > 0 for value in prices.values()):
                raise ValueError(f"Invalid OHLC value on CSV line {line_number}")
            if prices["low"] > min(prices["open"], prices["close"]):
                raise ValueError(f"Low is above open/close on CSV line {line_number}")
            if prices["high"] < max(prices["open"], prices["close"]):
                raise ValueError(f"High is below open/close on CSV line {line_number}")
            if prices["high"] < prices["low"]:
                raise ValueError(f"High is below low on CSV line {line_number}")
            if not math.isfinite(volume) or volume < 0:
                raise ValueError(f"Invalid volume on CSV line {line_number}")

            rows.append((pair, ts, prices["open"], prices["high"], prices["low"], prices["close"], volume))

    if not rows:
        raise ValueError("CSV contains no candles")
    rows.sort(key=lambda candle: candle[1])
    return source_hash, rows


def import_candles(csv_path, db_path):
    source_hash, rows = load_candles(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=60)

    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS candles(
                pair TEXT NOT NULL, ts INTEGER NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY(pair, ts)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS legacy_candle_imports(
                source_sha256 TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                inserted_count INTEGER NOT NULL,
                imported_at_utc TEXT NOT NULL
            )
        """)

        imported = connection.execute(
            "SELECT 1 FROM legacy_candle_imports WHERE source_sha256=?",
            (source_hash,)
        ).fetchone()
        if imported:
            connection.rollback()
            print("Legacy candles already imported; no changes made.")
            return 0

        conflicts = []
        for pair, ts, op, high, low, close, volume in rows:
            existing = connection.execute(
                "SELECT open, high, low, close, volume FROM candles WHERE pair=? AND ts=?",
                (pair, ts)
            ).fetchone()
            if existing is None:
                continue

            candidate = (op, high, low, close, volume)
            if any(
                not math.isclose(float(old or 0.0), new, rel_tol=0.0, abs_tol=PRICE_TOLERANCE)
                for old, new in zip(existing, candidate)
            ):
                conflicts.append(ts)
                if len(conflicts) == 5:
                    break

        if conflicts:
            connection.rollback()
            shown = ", ".join(str(ts) for ts in conflicts)
            raise ValueError(
                "Existing EURUSD candles differ from the import at timestamp(s): "
                f"{shown}. No candles were imported; review the data sources first."
            )

        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO candles(pair,ts,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            rows
        )
        inserted_count = connection.total_changes - before
        connection.execute(
            """
            INSERT INTO legacy_candle_imports(
                source_sha256, source_file, row_count, inserted_count, imported_at_utc
            ) VALUES(?,?,?,?,?)
            """,
            (
                source_hash,
                csv_path.name,
                len(rows),
                inserted_count,
                datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
        )
        connection.commit()
        print(
            f"Imported {inserted_count} new EURUSD candles "
            f"({len(rows) - inserted_count} identical existing rows left unchanged)."
        )
        print(f"Source SHA-256: {source_hash}")
        return inserted_count
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description="Safely import validated legacy EURUSD candles.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Source CSV file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Target engine SQLite database")
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")

    import_candles(args.csv, args.db)


if __name__ == "__main__":
    main()
