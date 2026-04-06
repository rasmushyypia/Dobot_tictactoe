from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def load_summary(replay_dir: Path) -> list[dict[str, Any]]:
    summary_path = replay_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{replay_dir} has no summary.json")
    data = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{summary_path} does not contain a summary list")
    return data


def index_by_record(summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in summary:
        record = str(row.get("record") or "")
        if record:
            indexed[record] = row
    return indexed


def status_rank(status: str | None) -> int:
    ranks = {
        "valid": 3,
        "match": 3,
        "skipped": 2,
        "invalid": 1,
        "mismatch": 1,
        "error": 0,
    }
    return ranks.get(str(status or "").lower(), -1)


def classify_change(left_status: str | None, right_status: str | None) -> str:
    left_rank = status_rank(left_status)
    right_rank = status_rank(right_status)
    if left_rank == right_rank:
        return "unchanged"
    if right_rank > left_rank:
        return "improved"
    if right_rank < left_rank:
        return "regressed"
    return "changed"


def make_output_dir(left_dir: Path, right_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    replay_root = left_dir.parent
    dirname = f"compare_{timestamp}_{left_dir.name}_vs_{right_dir.name}"
    path = replay_root / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def summarize(summary: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline_statuses = [str(row.get("pipeline_status") or "") for row in summary]
    total_duration = [float(row["total_duration_ms"]) for row in summary if row.get("total_duration_ms") is not None]
    stage1_tokens = [int(row["stage1_tokens_total"]) for row in summary if row.get("stage1_tokens_total") is not None]
    stage2_tokens = [int(row["stage2_tokens_total"]) for row in summary if row.get("stage2_tokens_total") is not None]
    return {
        "records": len(summary),
        "valid": sum(1 for status in pipeline_statuses if status == "valid"),
        "invalid": sum(1 for status in pipeline_statuses if status == "invalid"),
        "skipped": sum(1 for status in pipeline_statuses if status == "skipped"),
        "error": sum(1 for status in pipeline_statuses if status == "error"),
        "avg_total_duration_ms": average(total_duration),
        "avg_stage1_tokens": average([float(value) for value in stage1_tokens]),
        "avg_stage2_tokens": average([float(value) for value in stage2_tokens]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "record",
        "change",
        "left_pipeline_status",
        "right_pipeline_status",
        "left_stage1_status",
        "right_stage1_status",
        "left_stage2_status",
        "right_stage2_status",
        "left_total_duration_ms",
        "right_total_duration_ms",
        "left_stage1_tokens_total",
        "right_stage1_tokens_total",
        "left_stage2_tokens_total",
        "right_stage2_tokens_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two pipeline replay folders and report improvements/regressions per record.")
    parser.add_argument("--left", required=True, help="Left pipeline replay folder.")
    parser.add_argument("--right", required=True, help="Right pipeline replay folder.")
    parser.add_argument("--output-dir", help="Optional output directory for comparison artifacts.")
    args = parser.parse_args()

    left_dir = Path(args.left)
    right_dir = Path(args.right)
    output_dir = make_output_dir(left_dir, right_dir, Path(args.output_dir) if args.output_dir else None)

    left_summary = load_summary(left_dir)
    right_summary = load_summary(right_dir)
    left_index = index_by_record(left_summary)
    right_index = index_by_record(right_summary)

    all_records = sorted(set(left_index) | set(right_index))
    per_record: list[dict[str, Any]] = []
    for record in all_records:
        left_row = left_index.get(record)
        right_row = right_index.get(record)
        left_status = left_row.get("pipeline_status") if left_row else None
        right_status = right_row.get("pipeline_status") if right_row else None
        per_record.append(
            {
                "record": record,
                "change": classify_change(left_status, right_status),
                "left_pipeline_status": left_status,
                "right_pipeline_status": right_status,
                "left_stage1_status": left_row.get("stage1_status") if left_row else None,
                "right_stage1_status": right_row.get("stage1_status") if right_row else None,
                "left_stage2_status": left_row.get("stage2_status") if left_row else None,
                "right_stage2_status": right_row.get("stage2_status") if right_row else None,
                "left_total_duration_ms": left_row.get("total_duration_ms") if left_row else None,
                "right_total_duration_ms": right_row.get("total_duration_ms") if right_row else None,
                "left_stage1_tokens_total": left_row.get("stage1_tokens_total") if left_row else None,
                "right_stage1_tokens_total": right_row.get("stage1_tokens_total") if right_row else None,
                "left_stage2_tokens_total": left_row.get("stage2_tokens_total") if left_row else None,
                "right_stage2_tokens_total": right_row.get("stage2_tokens_total") if right_row else None,
            }
        )

    overview = {
        "left": {
            "replay_dir": str(left_dir),
            "summary": summarize(left_summary),
        },
        "right": {
            "replay_dir": str(right_dir),
            "summary": summarize(right_summary),
        },
        "comparison": {
            "records_compared": len(per_record),
            "improved": sum(1 for row in per_record if row["change"] == "improved"),
            "regressed": sum(1 for row in per_record if row["change"] == "regressed"),
            "unchanged": sum(1 for row in per_record if row["change"] == "unchanged"),
        },
    }

    (output_dir / "comparison.json").write_text(
        json.dumps({"overview": overview, "per_record": per_record}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "comparison.csv", per_record)

    print(f"Comparison saved to: {output_dir}")
    print(json.dumps(overview, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
