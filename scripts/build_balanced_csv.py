"""Build a 100-row stratified slice (50 PASS + 50 FAIL) from memory_eval_sample.csv.

The benchmark harness uses head-N sampling, so we pre-balance the file here.
Run once; output goes to benchmarks/data/test_100_balanced.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
SRC = BENCH_DIR / "data" / "memory_eval_sample.csv"
DST = BENCH_DIR / "data" / "test_100_balanced.csv"

PER_CLASS = 50


def main() -> None:
    with open(SRC, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "score" not in fieldnames:
            raise SystemExit(f"'score' column missing in {SRC}")
        pass_rows: list[dict] = []
        fail_rows: list[dict] = []
        for row in reader:
            score = (row.get("score") or "").strip()
            if score == "0" and len(pass_rows) < PER_CLASS:
                pass_rows.append(row)
            elif score == "1" and len(fail_rows) < PER_CLASS:
                fail_rows.append(row)
            if len(pass_rows) == PER_CLASS and len(fail_rows) == PER_CLASS:
                break

    if len(pass_rows) < PER_CLASS or len(fail_rows) < PER_CLASS:
        raise SystemExit(
            f"Not enough rows: got {len(pass_rows)} PASS + {len(fail_rows)} FAIL"
        )

    with open(DST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pass_rows)
        writer.writerows(fail_rows)

    print(f"Wrote {len(pass_rows) + len(fail_rows)} rows → {DST}")
    print(f"  PASS (score=0): {len(pass_rows)}")
    print(f"  FAIL (score=1): {len(fail_rows)}")


if __name__ == "__main__":
    main()
