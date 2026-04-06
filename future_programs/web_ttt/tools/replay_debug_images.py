from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEBUG_ROOT = PROJECT_ROOT / "debug_images"
DEFAULT_PROMPTS_DIR = DEFAULT_DEBUG_ROOT / "test_prompts"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Model returned an empty response.")
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw_text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not parse a JSON object from the model response.")


def extract_model_text(payload: dict[str, Any]) -> str:
    response_text = str(payload.get("response", "") or "").strip()
    if response_text:
        return response_text
    thinking_text = str(payload.get("thinking", "") or "").strip()
    if thinking_text:
        return thinking_text
    raise ValueError("Model returned an empty response.")


def normalize_board(value: Any) -> list[str] | None:
    if not isinstance(value, list) or len(value) != 9:
        return None
    normalized: list[str] = []
    for item in value:
        if item in ("", "X", "O"):
            normalized.append(item)
            continue
        if isinstance(item, str):
            stripped = item.strip().upper()
            if stripped in ("", ".", "_", "-"):
                normalized.append("")
                continue
            if stripped in ("X", "O"):
                normalized.append(stripped)
                continue
        return None
    return normalized


def format_board_rows(board: list[str]) -> str:
    rows = []
    for row_index in range(3):
        row = board[row_index * 3 : row_index * 3 + 3]
        rows.append(" ".join(cell if cell else "." for cell in row))
    return "\n".join(rows)


def mismatch_cells(source_board: list[str], interpreted_board: list[str]) -> list[str]:
    mismatches: list[str] = []
    for index, (source, interpreted) in enumerate(zip(source_board, interpreted_board, strict=False)):
        if source != interpreted:
            source_value = source if source else "."
            interpreted_value = interpreted if interpreted else "."
            mismatches.append(f"{index}:{source_value}->{interpreted_value}")
    return mismatches


def load_prompt_template(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_prompt(template: str, record: dict[str, Any]) -> str:
    source_board = [cell if cell in ("", "X", "O") else "" for cell in record["source_board"]]
    legal_moves = [index for index, cell in enumerate(source_board) if cell == ""]
    replacements = {
        "player": str(record.get("requested_player", "X")),
        "source_board_rows": format_board_rows(source_board),
        "source_board_list": json.dumps(source_board),
        "legal_moves": json.dumps(legal_moves),
        "observation_mode": str(record.get("observation_mode", "camera_frame")),
    }
    return template.format(**replacements)


def resolve_debug_records_dir(debug_dir: Path | None = None, dataset: str | None = None) -> Path:
    if debug_dir is not None:
        return debug_dir
    if dataset:
        return DEFAULT_DEBUG_ROOT / "datasets" / dataset / "records"
    return DEFAULT_DEBUG_ROOT


def make_output_dir(debug_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = debug_dir.parent / "replays" if debug_dir.name == "records" else debug_dir / "replays"
    output_dir = output_root / f"replay_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_record_paths(
    debug_dir: Path,
    limit: int,
    records: list[str],
    pattern: str | None,
) -> list[Path]:
    record_paths = sorted(debug_dir.glob("*.json"))
    if records:
        allowed = set(records)
        record_paths = [path for path in record_paths if path.name in allowed]
    if pattern:
        record_paths = [path for path in record_paths if pattern in path.name]
    if limit > 0:
        record_paths = record_paths[-limit:]
    return record_paths


def write_run_metadata(output_dir: Path, args: argparse.Namespace, prompt_template: str | None) -> None:
    metadata = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "debug_dir": args.debug_dir,
        "dataset": getattr(args, "dataset", None),
        "prompt_file": args.prompt_file,
        "model_override": args.model,
        "ollama_url": args.ollama_url,
        "limit": args.limit,
        "records": args.record,
        "pattern": args.pattern,
        "timeout_seconds": args.timeout_seconds,
        "retries": args.retries,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if prompt_template is not None:
        (output_dir / "prompt_used.txt").write_text(prompt_template, encoding="utf-8")


def write_summary_csv(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "record",
        "status",
        "matched_source_board",
        "mismatch_count",
        "mismatches",
        "duration_ms",
        "total_duration_ms",
        "prompt_eval_count",
        "eval_count",
        "tokens_total",
        "attempts",
        "model",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "record": row.get("record"),
                    "status": row.get("status"),
                    "matched_source_board": row.get("matched_source_board"),
                    "mismatch_count": row.get("mismatch_count"),
                    "mismatches": "; ".join(row.get("mismatches", [])),
                    "duration_ms": row.get("duration_ms"),
                    "total_duration_ms": row.get("total_duration_ms"),
                    "prompt_eval_count": row.get("prompt_eval_count"),
                    "eval_count": row.get("eval_count"),
                    "tokens_total": row.get("tokens_total"),
                    "attempts": row.get("attempts"),
                    "model": row.get("model"),
                    "error": row.get("error"),
                }
            )


def replay_record(
    record_path: Path,
    ollama_url: str,
    model: str | None,
    prompt_template: str | None,
    output_dir: Path,
    timeout_seconds: float,
    retries: int,
    num_predict: int | None,
    temperature: float,
) -> dict[str, Any]:
    record = load_json_file(record_path)
    image_path = record.get("image_path")
    if not image_path:
        raise ValueError(f"{record_path.name} has no image_path.")

    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    prompt = (
        build_prompt(prompt_template, record)
        if prompt_template is not None
        else str(record.get("prompt_preview", "")).strip()
    )
    if not prompt:
        raise ValueError(f"{record_path.name} has no prompt_preview and no prompt template was provided.")

    schema = {
        "type": "object",
        "properties": {
            "interpreted_board": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 9,
                "maxItems": 9,
            },
            "reasoning_transcript": {"type": "string"},
        },
        "required": ["interpreted_board", "reasoning_transcript"],
    }
    source_board = [cell if cell in ("", "X", "O") else "" for cell in record["source_board"]]
    output_path = output_dir / record_path.name
    start = time.perf_counter()
    attempts = 0
    last_error: str | None = None
    model_name = model or str(record["model"])

    while attempts <= retries:
        attempts += 1
        try:
            options: dict[str, Any] = {"temperature": temperature}
            if num_predict is not None:
                options["num_predict"] = num_predict
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema,
                    "images": [image_base64],
                    "options": options,
                },
                timeout=(3.0, timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))

            raw_text = extract_model_text(payload)
            parsed = extract_json_object(raw_text)
            interpreted_board = normalize_board(parsed.get("interpreted_board"))
            if interpreted_board is None:
                raise ValueError("Replay returned invalid interpreted_board.")

            prompt_eval_count = payload.get("prompt_eval_count")
            eval_count = payload.get("eval_count")
            total_duration_ns = payload.get("total_duration")
            try:
                prompt_eval_count = int(prompt_eval_count) if prompt_eval_count is not None else None
            except (TypeError, ValueError):
                prompt_eval_count = None
            try:
                eval_count = int(eval_count) if eval_count is not None else None
            except (TypeError, ValueError):
                eval_count = None
            try:
                total_duration_ms = round(float(total_duration_ns) / 1_000_000.0, 1) if total_duration_ns is not None else None
            except (TypeError, ValueError):
                total_duration_ms = None

            mismatches = mismatch_cells(source_board, interpreted_board)
            result = {
                "input_record": str(record_path),
                "image_path": image_path,
                "model": model_name,
                "prompt": prompt,
                "source_board": source_board,
                "interpreted_board": interpreted_board,
                "matched_source_board": len(mismatches) == 0,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "reasoning_transcript": str(parsed.get("reasoning_transcript", "")).strip(),
                "raw_response": payload,
                "status": "match" if len(mismatches) == 0 else "mismatch",
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                "total_duration_ms": total_duration_ms,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "tokens_total": (
                    (prompt_eval_count or 0) + (eval_count or 0)
                    if (prompt_eval_count is not None or eval_count is not None)
                    else None
                ),
                "attempts": attempts,
                "error": None,
            }
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempts > retries:
                break

    result = {
        "input_record": str(record_path),
        "image_path": image_path,
        "model": model_name,
        "prompt": prompt,
        "source_board": source_board,
        "interpreted_board": None,
        "matched_source_board": False,
        "mismatch_count": None,
        "mismatches": [],
        "reasoning_transcript": "",
        "raw_response": None,
        "status": "error",
        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
        "total_duration_ms": None,
        "prompt_eval_count": None,
        "eval_count": None,
        "tokens_total": None,
        "attempts": attempts,
        "error": last_error,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay saved web_ttt debug images against Ollama with a saved or custom prompt.")
    parser.add_argument(
        "--debug-dir",
        help="Directory containing saved debug image/json pairs.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name under debug_images/datasets/<dataset>/records.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional text file for a custom replay prompt. Placeholders: {player}, {source_board_rows}, {source_board_list}, {legal_moves}, {observation_mode}",
    )
    parser.add_argument(
        "--model",
        help="Optional Ollama model override. Defaults to the model stored in each debug record.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Replay only the latest N records. 0 means all records.",
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        help="Replay a specific debug record filename. Repeat for multiple records.",
    )
    parser.add_argument(
        "--pattern",
        help="Replay only records whose filename contains this substring.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Read timeout for each Ollama replay request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries per record after the initial attempt.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        help="Optional maximum output tokens requested from Ollama for each replay. By default, match the live GUI path and do not set num_predict.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for replay requests.",
    )
    args = parser.parse_args()

    debug_dir = resolve_debug_records_dir(Path(args.debug_dir) if args.debug_dir else None, args.dataset)
    prompt_template = load_prompt_template(Path(args.prompt_file)) if args.prompt_file else None
    record_paths = select_record_paths(debug_dir, args.limit, args.record, args.pattern)

    if not record_paths:
        print("No debug records found.", file=sys.stderr)
        return 1

    output_dir = make_output_dir(debug_dir)
    write_run_metadata(output_dir, args, prompt_template)
    summary: list[dict[str, Any]] = []
    total = len(record_paths)
    for index, record_path in enumerate(record_paths, start=1):
        print(
            f"[{index}/{total}] RUN       {record_path.name}  "
            f"(timeout={args.timeout_seconds}s, retries={args.retries})",
            flush=True,
        )
        result = replay_record(
            record_path=record_path,
            ollama_url=args.ollama_url,
            model=args.model,
            prompt_template=prompt_template,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            num_predict=args.num_predict,
            temperature=args.temperature,
        )
        summary_row = {
            "record": record_path.name,
            "status": result["status"],
            "matched_source_board": result["matched_source_board"],
            "mismatch_count": result["mismatch_count"],
            "mismatches": result["mismatches"],
            "duration_ms": result["duration_ms"],
            "total_duration_ms": result["total_duration_ms"],
            "prompt_eval_count": result["prompt_eval_count"],
            "eval_count": result["eval_count"],
            "tokens_total": result["tokens_total"],
            "attempts": result["attempts"],
            "model": result["model"],
            "error": result["error"],
        }
        summary.append(summary_row)
        prefix = f"[{index}/{total}]"
        if result["status"] == "match":
            print(f"{prefix} MATCH     {record_path.name}  ({result['duration_ms']} ms)")
        elif result["status"] == "mismatch":
            print(
                f"{prefix} MISMATCH  {record_path.name}  "
                f"({result['duration_ms']} ms, mismatches={result['mismatch_count']})"
            )
        else:
            print(
                f"{prefix} ERROR     {record_path.name}  "
                f"({result['duration_ms']} ms, attempts={result['attempts']}): {result['error']}",
                file=sys.stderr,
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(output_dir, summary)
    print(f"\nReplay results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
