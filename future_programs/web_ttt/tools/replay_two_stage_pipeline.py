from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from replay_debug_images import (
    DEFAULT_DEBUG_ROOT,
    DEFAULT_PROMPTS_DIR,
    extract_json_object,
    extract_model_text,
    format_board_rows,
    load_json_file,
    load_prompt_template,
    normalize_board,
    resolve_debug_records_dir,
    select_record_paths,
)


DEFAULT_DEBUG_DIR = DEFAULT_DEBUG_ROOT
DEFAULT_STAGE1_PROMPT = DEFAULT_PROMPTS_DIR / "camera_board_analysis_v8.txt"
DEFAULT_STAGE2_PROMPT = DEFAULT_PROMPTS_DIR / "move_reasoning_v1.txt"


def make_output_dir(debug_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = debug_dir.parent / "replays" if debug_dir.name == "records" else debug_dir / "replays"
    output_dir = output_root / f"pipeline_replay_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def legal_moves(board: list[str]) -> list[int]:
    return [index for index, cell in enumerate(board) if cell == ""]


def mismatch_cells(source_board: list[str], interpreted_board: list[str]) -> list[str]:
    mismatches: list[str] = []
    for index, (source, interpreted) in enumerate(zip(source_board, interpreted_board, strict=False)):
        if source != interpreted:
            source_value = source if source else "."
            interpreted_value = interpreted if interpreted else "."
            mismatches.append(f"{index}:{source_value}->{interpreted_value}")
    return mismatches


def extract_final_move_from_transcript(transcript: str) -> int | None:
    match = re.search(r"FINAL_MOVE:\s*(\d+)\s*$", transcript.strip(), re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def build_stage1_prompt(template: str, record: dict[str, Any]) -> str:
    source_board = [cell if cell in ("", "X", "O") else "" for cell in record["source_board"]]
    replacements = {
        "player": str(record.get("requested_player", "O")),
        "source_board_rows": format_board_rows(source_board),
        "source_board_list": json.dumps(source_board),
        "legal_moves": json.dumps(legal_moves(source_board)),
        "observation_mode": str(record.get("observation_mode", "camera_frame")),
    }
    return template.format(**replacements)


def build_stage2_prompt(template: str, interpreted_board: list[str]) -> str:
    replacements = {
        "player": "O",
        "interpreted_board_rows": format_board_rows(interpreted_board),
        "interpreted_board_list": json.dumps(interpreted_board),
        "legal_moves": json.dumps(legal_moves(interpreted_board)),
        "observation_mode": "two_stage_pipeline",
    }
    return template.format(**replacements)


def request_ollama_json(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    timeout_seconds: float,
    temperature: float,
    num_predict: int | None,
    image_base64: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    options: dict[str, Any] = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "images": [image_base64] if image_base64 else [],
            "options": options,
        },
        timeout=(3.0, timeout_seconds),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))

    parsed = extract_json_object(extract_model_text(payload))
    return parsed, payload


def token_stats(payload: dict[str, Any] | None) -> dict[str, int | float | None]:
    if payload is None:
        return {
            "prompt_eval_count": None,
            "eval_count": None,
            "tokens_total": None,
            "total_duration_ms": None,
        }

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

    return {
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "tokens_total": (
            (prompt_eval_count or 0) + (eval_count or 0)
            if (prompt_eval_count is not None or eval_count is not None)
            else None
        ),
        "total_duration_ms": total_duration_ms,
    }


def run_stage1(
    *,
    record: dict[str, Any],
    image_base64: str,
    prompt_template: str,
    model: str,
    ollama_url: str,
    timeout_seconds: float,
    temperature: float,
    num_predict: int | None,
) -> dict[str, Any]:
    source_board = [cell if cell in ("", "X", "O") else "" for cell in record["source_board"]]
    prompt = build_stage1_prompt(prompt_template, record)
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

    start = time.perf_counter()
    parsed, payload = request_ollama_json(
        ollama_url=ollama_url,
        model=model,
        prompt=prompt,
        schema=schema,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        num_predict=num_predict,
        image_base64=image_base64,
    )
    interpreted_board = normalize_board(parsed.get("interpreted_board"))
    if interpreted_board is None:
        raise ValueError("Stage 1 returned invalid interpreted_board.")

    mismatches = mismatch_cells(source_board, interpreted_board)
    result = {
        "model": model,
        "prompt": prompt,
        "interpreted_board": interpreted_board,
        "reasoning_transcript": str(parsed.get("reasoning_transcript", "")).strip(),
        "mismatches": mismatches,
        "matched_source_board": len(mismatches) == 0,
        "status": "match" if len(mismatches) == 0 else "mismatch",
        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
        "raw_response": payload,
    }
    result.update(token_stats(payload))
    return result


def run_stage2(
    *,
    interpreted_board: list[str],
    source_board: list[str],
    prompt_template: str,
    model: str,
    ollama_url: str,
    timeout_seconds: float,
    temperature: float,
    num_predict: int | None,
) -> dict[str, Any]:
    prompt = build_stage2_prompt(prompt_template, interpreted_board)
    stage2_legal_moves = legal_moves(interpreted_board)
    source_legal_moves = legal_moves(source_board)
    schema = {
        "type": "object",
        "properties": {
            "chosen_move": {"type": "integer"},
            "reasoning_transcript": {"type": "string"},
        },
        "required": ["chosen_move", "reasoning_transcript"],
    }

    start = time.perf_counter()
    parsed, payload = request_ollama_json(
        ollama_url=ollama_url,
        model=model,
        prompt=prompt,
        schema=schema,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        num_predict=num_predict,
        image_base64=None,
    )

    chosen_move = int(parsed["chosen_move"])
    reasoning_transcript = str(parsed.get("reasoning_transcript", "")).strip()
    transcript_move = extract_final_move_from_transcript(reasoning_transcript)

    legal_on_interpreted = chosen_move in stage2_legal_moves
    legal_on_source = chosen_move in source_legal_moves

    validation_error = None
    if not legal_on_interpreted:
        validation_error = f"Stage 2 selected illegal move {chosen_move} for interpreted board legal moves {stage2_legal_moves}."
    elif transcript_move is None:
        validation_error = "Stage 2 reasoning_transcript did not end with a valid FINAL_MOVE line."
    elif transcript_move != chosen_move:
        validation_error = f"Stage 2 transcript ended with FINAL_MOVE: {transcript_move}, but chosen_move was {chosen_move}."

    result = {
        "model": model,
        "prompt": prompt,
        "current_player": "O",
        "legal_moves_interpreted": stage2_legal_moves,
        "legal_moves_source": source_legal_moves,
        "chosen_move": chosen_move,
        "reasoning_transcript": reasoning_transcript,
        "legal_on_interpreted_board": legal_on_interpreted,
        "legal_on_source_board": legal_on_source,
        "transcript_final_move": transcript_move,
        "status": "valid" if validation_error is None else "invalid",
        "validation_error": validation_error,
        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
        "raw_response": payload,
    }
    result.update(token_stats(payload))
    return result


def write_summary_csv(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "record",
        "pipeline_status",
        "stage1_status",
        "stage1_mismatch_count",
        "stage2_status",
        "stage2_chosen_move",
        "stage2_legal_on_interpreted_board",
        "stage2_legal_on_source_board",
        "stage1_duration_ms",
        "stage2_duration_ms",
        "total_duration_ms",
        "stage1_tokens_total",
        "stage2_tokens_total",
        "stage1_model",
        "stage2_model",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_run_metadata(output_dir: Path, args: argparse.Namespace, stage1_prompt_template: str, stage2_prompt_template: str) -> None:
    metadata = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "debug_dir": args.debug_dir,
        "dataset": getattr(args, "dataset", None),
        "stage1_prompt_file": args.stage1_prompt_file,
        "stage2_prompt_file": args.stage2_prompt_file,
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "ollama_url": args.ollama_url,
        "limit": args.limit,
        "records": args.record,
        "pattern": args.pattern,
        "timeout_seconds": args.timeout_seconds,
        "stage1_num_predict": args.stage1_num_predict,
        "stage2_num_predict": args.stage2_num_predict,
        "stage1_temperature": args.stage1_temperature,
        "stage2_temperature": args.stage2_temperature,
        "assumed_player": "O",
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "stage1_prompt_used.txt").write_text(stage1_prompt_template, encoding="utf-8")
    (output_dir / "stage2_prompt_used.txt").write_text(stage2_prompt_template, encoding="utf-8")


def replay_pipeline_record(
    *,
    record_path: Path,
    output_dir: Path,
    stage1_prompt_template: str,
    stage2_prompt_template: str,
    stage1_model_override: str | None,
    stage2_model_override: str | None,
    ollama_url: str,
    timeout_seconds: float,
    stage1_temperature: float,
    stage2_temperature: float,
    stage1_num_predict: int | None,
    stage2_num_predict: int | None,
) -> dict[str, Any]:
    record = load_json_file(record_path)
    source_board = [cell if cell in ("", "X", "O") else "" for cell in record["source_board"]]
    image_path = record.get("image_path")
    start_total = time.perf_counter()
    output_path = output_dir / record_path.name
    result: dict[str, Any] = {
        "input_record": str(record_path),
        "image_path": image_path,
        "source_board": source_board,
        "pipeline_status": "error",
        "stage1": None,
        "stage2": None,
        "error": None,
        "duration_ms": None,
    }

    if not image_path:
        result["pipeline_status"] = "skipped"
        result["stage1"] = {
            "status": "skipped",
            "validation_error": "Record has no image_path. This replay only applies to image-based debug records.",
        }
        result["stage2"] = {
            "status": "skipped",
            "validation_error": "Record has no image_path. This replay only applies to image-based debug records.",
        }
        result["duration_ms"] = round((time.perf_counter() - start_total) * 1000, 1)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

    stage1_model = stage1_model_override or str(record.get("model") or "")
    if not stage1_model:
        raise ValueError(f"{record_path.name} has no model and no --stage1-model override.")
    stage2_model = stage2_model_override or stage1_model

    try:
        stage1 = run_stage1(
            record=record,
            image_base64=image_base64,
            prompt_template=stage1_prompt_template,
            model=stage1_model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
            temperature=stage1_temperature,
            num_predict=stage1_num_predict,
        )
        result["stage1"] = stage1

        interpreted_board = stage1["interpreted_board"]
        stage2_legal_moves = legal_moves(interpreted_board)
        if not stage2_legal_moves:
            stage2 = {
                "model": stage2_model,
                "prompt": None,
                "current_player": "O",
                "legal_moves_interpreted": [],
                "legal_moves_source": legal_moves(source_board),
                "chosen_move": None,
                "reasoning_transcript": "",
                "legal_on_interpreted_board": False,
                "legal_on_source_board": False,
                "transcript_final_move": None,
                "status": "skipped",
                "validation_error": "No legal moves remained on the interpreted board.",
                "duration_ms": 0.0,
                "raw_response": None,
                "prompt_eval_count": None,
                "eval_count": None,
                "tokens_total": None,
                "total_duration_ms": None,
            }
        else:
            stage2 = run_stage2(
                interpreted_board=interpreted_board,
                source_board=source_board,
                prompt_template=stage2_prompt_template,
                model=stage2_model,
                ollama_url=ollama_url,
                timeout_seconds=timeout_seconds,
                temperature=stage2_temperature,
                num_predict=stage2_num_predict,
            )
        result["stage2"] = stage2

        if stage2["status"] == "valid":
            result["pipeline_status"] = "valid"
        elif stage2["status"] == "skipped":
            result["pipeline_status"] = "skipped"
        else:
            result["pipeline_status"] = "invalid"
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    finally:
        result["duration_ms"] = round((time.perf_counter() - start_total) * 1000, 1)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a two-stage replay pipeline: image interpretation first, then legal O-move reasoning.")
    parser.add_argument("--debug-dir", help="Directory containing saved debug image/json pairs.")
    parser.add_argument("--dataset", help="Dataset name under debug_images/datasets/<dataset>/records.")
    parser.add_argument("--stage1-prompt-file", default=str(DEFAULT_STAGE1_PROMPT), help="Stage 1 image-to-board prompt template.")
    parser.add_argument("--stage2-prompt-file", default=str(DEFAULT_STAGE2_PROMPT), help="Stage 2 board-to-move prompt template.")
    parser.add_argument("--stage1-model", help="Optional stage 1 model override. Defaults to the model stored in each debug record.")
    parser.add_argument("--stage2-model", help="Optional stage 2 model override. Defaults to the stage 1 model.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--limit", type=int, default=0, help="Replay only the latest N records. 0 means all records.")
    parser.add_argument("--record", action="append", default=[], help="Replay a specific debug record filename. Repeat for multiple records.")
    parser.add_argument("--pattern", help="Replay only records whose filename contains this substring.")
    parser.add_argument("--timeout-seconds", type=float, default=45.0, help="Read timeout for each Ollama request.")
    parser.add_argument("--stage1-num-predict", type=int, help="Optional max output tokens for stage 1.")
    parser.add_argument("--stage2-num-predict", type=int, help="Optional max output tokens for stage 2.")
    parser.add_argument("--stage1-temperature", type=float, default=0.0, help="Sampling temperature for stage 1.")
    parser.add_argument("--stage2-temperature", type=float, default=0.0, help="Sampling temperature for stage 2.")
    args = parser.parse_args()

    debug_dir = resolve_debug_records_dir(Path(args.debug_dir) if args.debug_dir else None, args.dataset)
    stage1_prompt_template = load_prompt_template(Path(args.stage1_prompt_file))
    stage2_prompt_template = load_prompt_template(Path(args.stage2_prompt_file))
    if stage1_prompt_template is None:
        print("Stage 1 prompt file could not be loaded.", file=sys.stderr)
        return 1
    if stage2_prompt_template is None:
        print("Stage 2 prompt file could not be loaded.", file=sys.stderr)
        return 1

    record_paths = select_record_paths(debug_dir, args.limit, args.record, args.pattern)
    if not record_paths:
        print("No debug records found.", file=sys.stderr)
        return 1

    output_dir = make_output_dir(debug_dir)
    write_run_metadata(output_dir, args, stage1_prompt_template, stage2_prompt_template)
    summary: list[dict[str, Any]] = []

    total = len(record_paths)
    for index, record_path in enumerate(record_paths, start=1):
        print(f"[{index}/{total}] RUN       {record_path.name}", flush=True)
        result = replay_pipeline_record(
            record_path=record_path,
            output_dir=output_dir,
            stage1_prompt_template=stage1_prompt_template,
            stage2_prompt_template=stage2_prompt_template,
            stage1_model_override=args.stage1_model,
            stage2_model_override=args.stage2_model,
            ollama_url=args.ollama_url,
            timeout_seconds=args.timeout_seconds,
            stage1_temperature=args.stage1_temperature,
            stage2_temperature=args.stage2_temperature,
            stage1_num_predict=args.stage1_num_predict,
            stage2_num_predict=args.stage2_num_predict,
        )

        stage1 = result.get("stage1") or {}
        stage2 = result.get("stage2") or {}
        summary_row = {
            "record": record_path.name,
            "pipeline_status": result.get("pipeline_status"),
            "stage1_status": stage1.get("status"),
            "stage1_mismatch_count": len(stage1.get("mismatches", [])) if stage1 else None,
            "stage2_status": stage2.get("status"),
            "stage2_chosen_move": stage2.get("chosen_move"),
            "stage2_legal_on_interpreted_board": stage2.get("legal_on_interpreted_board"),
            "stage2_legal_on_source_board": stage2.get("legal_on_source_board"),
            "stage1_duration_ms": stage1.get("duration_ms"),
            "stage2_duration_ms": stage2.get("duration_ms"),
            "total_duration_ms": result.get("duration_ms"),
            "stage1_tokens_total": stage1.get("tokens_total"),
            "stage2_tokens_total": stage2.get("tokens_total"),
            "stage1_model": stage1.get("model"),
            "stage2_model": stage2.get("model"),
            "error": result.get("error") or stage2.get("validation_error"),
        }
        summary.append(summary_row)

        prefix = f"[{index}/{total}]"
        if result.get("pipeline_status") == "valid":
            print(
                f"{prefix} VALID     {record_path.name}  "
                f"(stage1={stage1.get('status')}, move={stage2.get('chosen_move')}, total={result.get('duration_ms')} ms)"
            )
        elif result.get("pipeline_status") == "invalid":
            print(
                f"{prefix} INVALID   {record_path.name}  "
                f"(stage1={stage1.get('status')}, move={stage2.get('chosen_move')}, reason={stage2.get('validation_error')})"
            )
        elif result.get("pipeline_status") == "skipped":
            print(
                f"{prefix} SKIPPED   {record_path.name}  "
                f"(stage1={stage1.get('status')}, reason={stage2.get('validation_error')})"
            )
        else:
            print(
                f"{prefix} ERROR     {record_path.name}  "
                f"({result.get('duration_ms')} ms): {result.get('error')}",
                file=sys.stderr,
            )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(output_dir, summary)
    print(f"\nPipeline replay results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
