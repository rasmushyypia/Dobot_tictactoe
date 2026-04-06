from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


Player = Literal["X", "O"]
Cell = Literal["", "X", "O"]
ObservationMode = Literal["direct_state", "camera_frame"]

OLLAMA_URL = os.getenv("WEB_TTT_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("WEB_TTT_OLLAMA_MODEL", "gemma4:e4b")
DEBUG_ROOT_DIR = Path(os.getenv("WEB_TTT_DEBUG_IMAGE_DIR", r"C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images"))
DEBUG_DATASET = os.getenv("WEB_TTT_DEBUG_DATASET", "").strip()
CAMERA_FRAME_ASPECT_WIDTH = int(os.getenv("WEB_TTT_CAMERA_ASPECT_WIDTH", "1"))
CAMERA_FRAME_ASPECT_HEIGHT = int(os.getenv("WEB_TTT_CAMERA_ASPECT_HEIGHT", "1"))
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ANALYSIS_PROMPT_FILE = PROJECT_ROOT / "debug_images" / "test_prompts" / "camera_board_analysis_v8.txt"
ANALYSIS_PROMPT_FILE = Path(os.getenv("WEB_TTT_ANALYSIS_PROMPT_FILE", str(DEFAULT_ANALYSIS_PROMPT_FILE)))
DEFAULT_MOVE_PROMPT_FILE = PROJECT_ROOT / "debug_images" / "test_prompts" / "move_reasoning_v1.txt"
MOVE_PROMPT_FILE = Path(os.getenv("WEB_TTT_MOVE_PROMPT_FILE", str(DEFAULT_MOVE_PROMPT_FILE)))

WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class MoveRequest(BaseModel):
    board: list[Cell] = Field(min_length=9, max_length=9)
    player: Player
    provider: Literal["mock", "ollama"] = "mock"
    model: str | None = None
    stage1_model: str | None = None
    stage2_model: str | None = None
    stage1_prompt_override: str | None = None
    stage2_prompt_override: str | None = None
    observation_mode: ObservationMode = "direct_state"
    analysis_only: bool = False


class AssistantResponse(BaseModel):
    provider: str
    model: str
    observation_model: str | None = None
    move_model: str | None = None
    debug_dataset: str | None = None
    current_player: Player
    legal_moves: list[int]
    interpreted_legal_moves: list[int] | None = None
    chosen_move: int | None
    proposed_move: int | None = None
    interpreted_board: list[Cell] | None = None
    reasoning_transcript: str
    observation_reasoning_transcript: str | None = None
    move_reasoning_transcript: str | None = None
    explanation: str
    confidence: float | None = None
    validation_status: Literal["valid", "invalid"] = "valid"
    validation_error: str | None = None
    prompt_preview: str | None = None
    observation_prompt_preview: str | None = None
    move_prompt_preview: str | None = None
    debug_image_path: str | None = None
    debug_record_path: str | None = None


class VisionStatus(BaseModel):
    active_source: Literal["camera", "synthetic"]
    camera_index: int
    streaming: bool
    note: str


class VisionConfigRequest(BaseModel):
    camera_index: int = Field(ge=0, le=10)


class ProviderOption(BaseModel):
    id: str
    label: str
    available: bool
    default_model: str | None = None
    note: str


class ProviderCatalog(BaseModel):
    default_provider: str
    providers: list[ProviderOption]


class PromptConfig(BaseModel):
    stage1_prompt: str
    stage2_prompt: str
    stage1_prompt_file: str | None = None
    stage2_prompt_file: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    provider: Literal["mock", "ollama"] = "mock"
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    provider: str
    model: str
    reply: str


def get_debug_records_dir() -> Path:
    if DEBUG_DATASET:
        return DEBUG_ROOT_DIR / "datasets" / DEBUG_DATASET / "records"
    return DEBUG_ROOT_DIR


class CameraService:
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._last_source: Literal["camera", "synthetic"] = "synthetic"

    def _ensure_capture(self) -> cv2.VideoCapture | None:
        if self._capture is None:
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not capture.isOpened():
                capture.release()
                return None
            self._capture = capture
        return self._capture

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_frame(self) -> np.ndarray:
        with self._lock:
            frame = self._read_raw_frame_locked()
            if frame is not None:
                return self._prepare_observation_frame(frame)

            self._last_source = "synthetic"
            return self._prepare_observation_frame(self._synthetic_frame())

    def capture_model_frame_base64(self) -> str:
        with self._lock:
            frame = self._read_raw_frame_locked()
            if frame is None:
                frame = self._prepare_observation_frame(self._synthetic_frame())
                self._last_source = "synthetic"
            else:
                frame = self._prepare_observation_frame(frame)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("Could not encode the current camera frame.")
            return base64.b64encode(buffer.tobytes()).decode("ascii")

    def status(self) -> VisionStatus:
        source = self._last_source
        note = (
            "Live camera frame is available."
            if source == "camera"
            else "No camera frame yet. Serving synthetic preview fallback."
        )
        return VisionStatus(
            active_source=source,
            camera_index=self.camera_index,
            streaming=True,
            note=note,
        )

    def close(self) -> None:
        with self._lock:
            self._release_capture()

    def set_camera_index(self, camera_index: int) -> None:
        with self._lock:
            if self.camera_index != camera_index:
                self.camera_index = camera_index
                self._last_source = "synthetic"
                self._release_capture()

    def _read_raw_frame_locked(self) -> np.ndarray | None:
        capture = self._ensure_capture()
        if capture is None:
            return None
        ok, frame = capture.read()
        if ok and frame is not None:
            self._last_source = "camera"
            return frame
        self._release_capture()
        return None

    def _annotate_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def _prepare_observation_frame(self, frame: np.ndarray) -> np.ndarray:
        return self._crop_to_aspect(frame, CAMERA_FRAME_ASPECT_WIDTH, CAMERA_FRAME_ASPECT_HEIGHT)

    def _crop_to_aspect(self, frame: np.ndarray, aspect_width: int, aspect_height: int) -> np.ndarray:
        if aspect_width <= 0 or aspect_height <= 0:
            return frame

        height, width = frame.shape[:2]
        target_ratio = aspect_width / aspect_height
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.001:
            return frame

        if current_ratio > target_ratio:
            cropped_width = max(1, int(height * target_ratio))
            offset_x = max(0, (width - cropped_width) // 2)
            return frame[:, offset_x : offset_x + cropped_width]

        cropped_height = max(1, int(width / target_ratio))
        offset_y = max(0, (height - cropped_height) // 2)
        return frame[offset_y : offset_y + cropped_height, :]

    def _synthetic_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (26, 34, 39)
        gradient = np.linspace(0, 80, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.maximum(frame[:, :, 1], gradient)
        frame[:, :, 2] = np.maximum(frame[:, :, 2], gradient[::-1])
        left = self.width // 2 - 180
        top = self.height // 2 - 180
        cell = 120
        for offset in range(1, 3):
            x = left + offset * cell
            y = top + offset * cell
            cv2.line(frame, (x, top), (x, top + cell * 3), (238, 230, 215), 3)
            cv2.line(frame, (left, y), (left + cell * 3, y), (238, 230, 215), 3)
        cv2.putText(frame, "web_ttt vision preview", (36, 64), cv2.FONT_HERSHEY_DUPLEX, 1.1, (246, 238, 228), 2, cv2.LINE_AA)
        cv2.putText(frame, "No live camera detected. Synthetic board feed active.", (36, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 223, 219), 2, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("Generated %H:%M:%S"), (36, self.height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 223, 219), 2, cv2.LINE_AA)
        return frame


def board_winner(board: list[Cell]) -> Cell | None:
    for a, b, c in WINNING_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def legal_moves(board: list[Cell]) -> list[int]:
    return [index for index, cell in enumerate(board) if cell == ""]


def format_board_for_prompt(board: list[Cell]) -> str:
    rows = []
    for row_index in range(3):
        row = board[row_index * 3 : row_index * 3 + 3]
        rows.append(" ".join(cell if cell else "." for cell in row))
    return "\n".join(rows)


def load_analysis_prompt_template() -> str | None:
    try:
        if ANALYSIS_PROMPT_FILE.is_file():
            return ANALYSIS_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def load_move_prompt_template() -> str | None:
    try:
        if MOVE_PROMPT_FILE.is_file():
            return MOVE_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def current_prompt_config() -> PromptConfig:
    return PromptConfig(
        stage1_prompt=load_analysis_prompt_template() or "",
        stage2_prompt=load_move_prompt_template() or "",
        stage1_prompt_file=str(ANALYSIS_PROMPT_FILE) if ANALYSIS_PROMPT_FILE else None,
        stage2_prompt_file=str(MOVE_PROMPT_FILE) if MOVE_PROMPT_FILE else None,
    )


def build_analysis_prompt_from_template(
    template: str,
    request: MoveRequest,
    moves: list[int],
    image_payload: str | None,
) -> str:
    source_board_rows = format_board_for_prompt(request.board)
    replacements = {
        "player": request.player,
        "legal_moves": json.dumps(moves),
        "source_board_rows": source_board_rows,
        "source_board_list": json.dumps(request.board),
        "observation_mode": request.observation_mode,
    }
    prompt = template.format(**replacements)

    if request.observation_mode == "camera_frame" and image_payload:
        prompt += "\nInput source: live camera frame."
    else:
        prompt += "\nInput source: direct GUI board state."
        prompt += f"\nRaw board list: {request.board}"
    return prompt


def build_move_prompt_from_template(template: str, interpreted_board: list[Cell], current_player: Player = "O") -> str:
    replacements = {
        "player": current_player,
        "interpreted_board_rows": format_board_for_prompt(interpreted_board),
        "interpreted_board_list": json.dumps(interpreted_board),
        "legal_moves": json.dumps(legal_moves(interpreted_board)),
        "observation_mode": "two_stage_live_pipeline",
    }
    return template.format(**replacements)


def validate_board(board: list[Cell], player: Player, analysis_only: bool = False) -> None:
    if len(board) != 9:
        raise HTTPException(status_code=422, detail="Board must contain exactly 9 cells.")
    if any(cell not in ("", "X", "O") for cell in board):
        raise HTTPException(status_code=422, detail="Board contains invalid cell values.")

    x_count = sum(cell == "X" for cell in board)
    o_count = sum(cell == "O" for cell in board)
    if x_count < o_count or x_count - o_count > 1:
        raise HTTPException(status_code=422, detail="Board state is not reachable in a legal game.")

    winner = board_winner(board)
    if winner is not None and not analysis_only:
        raise HTTPException(status_code=409, detail=f"Game is already over. Winner: {winner}.")

    expected = "X" if x_count == o_count else "O"
    if player != expected and not analysis_only:
        raise HTTPException(status_code=422, detail=f"Expected player {expected} based on board counts, but received {player}.")


def choose_mock_move(board: list[Cell], player: Player) -> tuple[int, str]:
    opponent: Player = "O" if player == "X" else "X"
    moves = legal_moves(board)
    for move in moves:
        next_board = board.copy()
        next_board[move] = player
        if board_winner(next_board) == player:
            return move, f"I can win immediately by placing {player} at cell {move}."
    for move in moves:
        next_board = board.copy()
        next_board[move] = opponent
        if board_winner(next_board) == opponent:
            return move, f"I need to block {opponent} at cell {move} to prevent a loss."
    if 4 in moves:
        return 4, "The center is open, so I take it to maximize control."
    for move in (0, 2, 6, 8):
        if move in moves:
            return move, f"The center is unavailable, so I take corner {move}."
    move = moves[0]
    return move, f"No tactical pattern dominates the position, so I take legal move {move}."


def clamp_confidence(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, numeric))


def normalize_interpreted_board(value: object) -> list[Cell] | None:
    if not isinstance(value, list) or len(value) != 9:
        return None

    normalized: list[Cell] = []
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


def extract_final_move_from_transcript(transcript: str) -> int | None:
    match = re.search(r"FINAL_MOVE:\s*(\d+)\s*$", transcript.strip(), re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_json_object(raw_text: str) -> dict[str, object]:
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


def extract_model_text(payload: dict[str, object]) -> str:
    response_text = str(payload.get("response", "") or "").strip()
    if response_text:
        return response_text
    raise ValueError("Model returned an empty response.")


def describe_board_mismatch(source_board: list[Cell], interpreted_board: list[Cell]) -> str | None:
    mismatches: list[str] = []
    for index, (source, interpreted) in enumerate(zip(source_board, interpreted_board, strict=False)):
        if source != interpreted:
            source_value = source if source else "."
            interpreted_value = interpreted if interpreted else "."
            mismatches.append(f"{index}:{source_value}->{interpreted_value}")
    if not mismatches:
        return None
    return "Board interpretation mismatch at cells: " + ", ".join(mismatches)


def make_debug_run_id(observation_mode: ObservationMode, analysis_only: bool) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    mode = "analysis" if analysis_only else "move"
    return f"web_ttt_{timestamp}_{millis:03d}_{observation_mode}_{mode}"


def save_debug_image(image_payload: str | None, observation_mode: ObservationMode, run_id: str) -> str | None:
    if not image_payload:
        return None

    image_bytes = base64.b64decode(image_payload)
    records_dir = get_debug_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    extension = ".jpg"
    filename = f"{run_id}{extension}"
    path = records_dir / filename
    with path.open("wb") as handle:
        handle.write(image_bytes)
    return str(path)


def save_debug_record(
    run_id: str,
    request: MoveRequest,
    model_name: str,
    image_path: str | None,
    response: AssistantResponse,
) -> str:
    records_dir = get_debug_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    record_path = records_dir / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "debug_dataset": DEBUG_DATASET or None,
        "provider": request.provider,
        "model": model_name,
        "stage1_model_requested": request.stage1_model,
        "stage2_model_requested": request.stage2_model,
        "observation_mode": request.observation_mode,
        "analysis_only": request.analysis_only,
        "requested_player": request.player,
        "source_board": request.board,
        "image_path": image_path,
        "observation_model": response.observation_model,
        "move_model": response.move_model,
        "current_player": response.current_player,
        "legal_moves": response.legal_moves,
        "chosen_move": response.chosen_move,
        "proposed_move": response.proposed_move,
        "interpreted_board": response.interpreted_board,
        "interpreted_legal_moves": response.interpreted_legal_moves,
        "reasoning_transcript": response.reasoning_transcript,
        "observation_reasoning_transcript": response.observation_reasoning_transcript,
        "move_reasoning_transcript": response.move_reasoning_transcript,
        "explanation": response.explanation,
        "confidence": response.confidence,
        "validation_status": response.validation_status,
        "validation_error": response.validation_error,
        "prompt_preview": response.prompt_preview,
        "observation_prompt_preview": response.observation_prompt_preview,
        "move_prompt_preview": response.move_prompt_preview,
    }
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(record_path)


def attach_debug_record(
    response: AssistantResponse,
    run_id: str,
    request: MoveRequest,
    model_name: str,
    image_path: str | None,
) -> AssistantResponse:
    response.debug_dataset = DEBUG_DATASET or None
    response.debug_image_path = image_path
    response.debug_record_path = save_debug_record(run_id, request, model_name, image_path, response)
    return response


def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return response.ok
    except requests.RequestException:
        return False


def list_provider_options() -> ProviderCatalog:
    ollama_is_up = ollama_available()
    return ProviderCatalog(
        default_provider="ollama" if ollama_is_up else "mock",
        providers=[
            ProviderOption(id="mock", label="Mock Strategist", available=True, default_model=None, note="Deterministic fallback for UI and validation work."),
            ProviderOption(
                id="ollama",
                label="Ollama Local Model",
                available=ollama_is_up,
                default_model=DEFAULT_OLLAMA_MODEL,
                note=(f"Uses {OLLAMA_URL} with model {DEFAULT_OLLAMA_MODEL}." if ollama_is_up else f"Ollama is not reachable at {OLLAMA_URL}."),
            ),
        ],
    )


def build_mock_response(request: MoveRequest, moves: list[int]) -> AssistantResponse:
    run_id = make_debug_run_id(request.observation_mode, request.analysis_only)
    if request.analysis_only:
        observed_state = ", ".join(
            f"{index}={cell if cell else '.'}" for index, cell in enumerate(request.board)
        )
        transcript = "\n".join(
            (
                f"OBSERVED_STATE: {observed_state}",
                f"X_CELLS: {[index for index, cell in enumerate(request.board) if cell == 'X']}",
                f"O_CELLS: {[index for index, cell in enumerate(request.board) if cell == 'O']}",
                f"EMPTY_CELLS: {moves}",
            )
        )
        return attach_debug_record(
            AssistantResponse(
            provider=request.provider,
            model="mock-strategist-v1",
            observation_model="mock-strategist-v1",
            move_model=None,
            current_player=request.player,
            legal_moves=moves,
            chosen_move=None,
            proposed_move=None,
            interpreted_board=request.board.copy(),
            reasoning_transcript=transcript,
            explanation="Observation only. No move requested.",
            validation_status="valid",
            prompt_preview="Mock analysis mode uses the GUI board state directly and does not perform image interpretation.",
            ),
            run_id=run_id,
            request=request,
            model_name="mock-strategist-v1",
            image_path=None,
        )

    chosen_move, rationale = choose_mock_move(request.board, request.player)
    transcript = "\n".join(
        (
            f"Provider: {request.provider}",
            f"Observation mode: {request.observation_mode}",
            (
                "Observed input: live camera frame"
                if request.observation_mode == "camera_frame"
                else f"Observed board:\n{format_board_for_prompt(request.board)}"
            ),
            f"Current player: {request.player}",
            f"Legal moves: {moves}",
            rationale,
            f"FINAL_MOVE: {chosen_move}",
        )
    )
    return attach_debug_record(
        AssistantResponse(
            provider=request.provider,
            model="mock-strategist-v1",
            observation_model="mock-strategist-v1",
            move_model="mock-strategist-v1",
            current_player=request.player,
            legal_moves=moves,
            chosen_move=chosen_move,
            proposed_move=chosen_move,
            interpreted_board=request.board.copy(),
            reasoning_transcript=transcript,
            explanation=f"Validated structured move: cell {chosen_move}.",
            confidence=0.68,
            validation_status="valid",
            prompt_preview="Mock move mode uses deterministic rule-based play on the GUI board state.",
        ),
        run_id=run_id,
        request=request,
        model_name="mock-strategist-v1",
        image_path=None,
    )


def build_ollama_response(request: MoveRequest, moves: list[int]) -> AssistantResponse:
    observation_model_name = request.stage1_model or request.model or DEFAULT_OLLAMA_MODEL
    move_model_name = request.stage2_model or request.model or observation_model_name
    run_id = make_debug_run_id(request.observation_mode, request.analysis_only)
    image_payload = None
    if request.observation_mode == "camera_frame":
        try:
            image_payload = camera_service.capture_model_frame_base64()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    debug_image_path = save_debug_image(image_payload, request.observation_mode, run_id)

    if request.observation_mode == "direct_state":
        interpreted_board = request.board.copy()
        interpreted_moves = moves.copy()
        observation_reasoning_transcript = "\n".join(
            (
                f"OBSERVED_STATE: {''.join(cell if cell else '.' for cell in interpreted_board)}",
                f"X_CELLS: {[index for index, cell in enumerate(interpreted_board) if cell == 'X']}",
                f"O_CELLS: {[index for index, cell in enumerate(interpreted_board) if cell == 'O']}",
                f"EMPTY_CELLS: {interpreted_moves}",
            )
        )
        stage1_prompt = "Direct state mode bypasses image interpretation and mirrors the GUI board exactly."
        mismatch_error = None
    else:
        interpreted_board = None
        interpreted_moves = None
        observation_reasoning_transcript = ""
        stage1_prompt = ""
        mismatch_error = None

    stage1_schema = {
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

    if interpreted_board is None:
        analysis_template = (request.stage1_prompt_override or "").strip() or load_analysis_prompt_template()
        if analysis_template:
            try:
                stage1_prompt = build_analysis_prompt_from_template(analysis_template, request, moves, image_payload)
            except (KeyError, ValueError):
                analysis_template = None

        if not analysis_template:
            prompt_parts = [
                "You are the vision-analysis assistant in a robotics lab tic-tac-toe demo.",
                f"The GUI currently says it is player {request.player}'s turn.",
                "Your only job is to infer the board state from the provided input.",
                "Do not suggest or execute a move.",
                "Do not chat. Do not self-correct repeatedly. Do not narrate uncertainty at length.",
                "",
                "--- SPATIAL REFERENCE ---",
                "The board is a 3x3 grid indexed 0 to 8 in row-major order:",
                "0: Top-Left    | 1: Top-Middle    | 2: Top-Right",
                "3: Middle-Left | 4: Center        | 5: Middle-Right",
                "6: Bottom-Left | 7: Bottom-Middle | 8: Bottom-Right",
                "",
                "--- OUTPUT REQUIREMENTS ---",
                "Return JSON only.",
                "reasoning_transcript must contain exactly these 4 lines:",
                "1. OBSERVED_STATE: <describe cells 0-8 using X O .>",
                "2. X_CELLS: <python-style list>",
                "3. O_CELLS: <python-style list>",
                "4. EMPTY_CELLS: <python-style list>",
            ]
            if request.observation_mode == "camera_frame" and image_payload:
                prompt_parts.extend(
                    (
                        "",
                        "--- VISION TASK ---",
                        "Look at the attached camera image and identify the 3x3 tic-tac-toe grid.",
                        "Map the physical marks (X, O, or empty) to the 0-8 spatial map provided above.",
                        "If the image is blurry or ambiguous, favor the 'empty' state for that cell.",
                        "Verify your detected state against the player turn (if it is O's turn, there should typically be an equal number of X and O, or one more X).",
                        f"Legal moves are: {moves}",
                        "Return interpreted_board as an array of 9 strings using '', 'X', or 'O'.",
                    )
                )
            else:
                prompt_parts.extend(
                    (
                        "",
                        "--- DIRECT STATE TASK ---",
                        f"Board as rows:\n{format_board_for_prompt(request.board)}",
                        f"Raw board list: {request.board}",
                        f"Legal moves: {moves}",
                        "Set interpreted_board equal to the current board.",
                    )
                )
            prompt_parts.extend(
                (
                    "",
                    "--- JSON SCHEMA ---",
                    "- interpreted_board: array of 9 strings",
                    "- reasoning_transcript: exactly 4 lines in the required format",
                )
            )
            stage1_prompt = "\n".join(prompt_parts)

    def request_ollama(
        prompt: str,
        schema: dict[str, object],
        model_name: str,
        include_image: bool = True,
    ) -> dict[str, object]:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "format": schema,
                "images": [image_payload] if include_image and image_payload and request.observation_mode == "camera_frame" else [],
                "options": {"temperature": 0.0},
            },
            timeout=(3.0, 90.0),
        )
        response.raise_for_status()
        return response.json()

    if interpreted_board is None:
        try:
            stage1_payload = request_ollama(stage1_prompt, stage1_schema, observation_model_name, include_image=True)
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc
        if stage1_payload.get("error"):
            raise HTTPException(status_code=502, detail=f"Ollama error: {stage1_payload['error']}")

        try:
            stage1_parsed = extract_json_object(extract_model_text(stage1_payload))
            interpreted_board = normalize_interpreted_board(stage1_parsed.get("interpreted_board"))
            observation_reasoning_transcript = str(stage1_parsed.get("reasoning_transcript", "")).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Ollama returned invalid structured output: {exc}") from exc

        if interpreted_board is None:
            raise HTTPException(status_code=502, detail="Ollama returned an invalid interpreted_board.")

        interpreted_moves = legal_moves(interpreted_board)
        mismatch_error = describe_board_mismatch(request.board, interpreted_board)

    if request.analysis_only:
        validation_status: Literal["valid", "invalid"] = "invalid" if mismatch_error else "valid"
        explanation = (
            "Board interpretation matches the known source board."
            if mismatch_error is None
            else "Board interpretation does not match the known source board."
        )
        return attach_debug_record(
            AssistantResponse(
                provider=request.provider,
                model=observation_model_name,
                observation_model=observation_model_name,
                move_model=None,
                current_player=request.player,
                legal_moves=moves,
                interpreted_legal_moves=interpreted_moves,
                chosen_move=None,
                proposed_move=None,
                interpreted_board=interpreted_board,
                reasoning_transcript=observation_reasoning_transcript,
                observation_reasoning_transcript=observation_reasoning_transcript,
                explanation=explanation,
                validation_status=validation_status,
                validation_error=mismatch_error,
                prompt_preview=stage1_prompt,
                observation_prompt_preview=stage1_prompt,
            ),
            run_id=run_id,
            request=request,
            model_name=observation_model_name,
            image_path=debug_image_path,
        )

    move_template = (request.stage2_prompt_override or "").strip() or load_move_prompt_template()
    if move_template:
        try:
            stage2_prompt = build_move_prompt_from_template(move_template, interpreted_board, current_player="O")
        except (KeyError, ValueError):
            move_template = None

    if not move_template:
        stage2_prompt = "\n".join(
            (
                "You are a tic-tac-toe move assistant.",
                "You always play as O.",
                "Your input board has already been interpreted from an image.",
                "You only need to choose a legal move for O.",
                f"Interpreted Board Rows:\n{format_board_for_prompt(interpreted_board)}",
                f"Interpreted Board List: {json.dumps(interpreted_board)}",
                f"Legal Moves: {interpreted_moves}",
                "Return JSON only.",
                "reasoning_transcript must contain exactly these 4 lines:",
                "BOARD: ...",
                "LEGAL_MOVES: <list>",
                "PLAN: <short note>",
                "FINAL_MOVE: <index>",
                "chosen_move must be one of Legal Moves.",
            )
        )

    stage2_schema = {
        "type": "object",
        "properties": {
            "chosen_move": {"type": "integer"},
            "reasoning_transcript": {"type": "string"},
        },
        "required": ["chosen_move", "reasoning_transcript"],
    }

    try:
        stage2_payload = request_ollama(stage2_prompt, stage2_schema, move_model_name, include_image=False)
        stage2_parsed = extract_json_object(extract_model_text(stage2_payload))
        chosen_move = int(stage2_parsed["chosen_move"])
        move_reasoning_transcript = str(stage2_parsed.get("reasoning_transcript", "")).strip()
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama returned invalid stage 2 output: {exc}") from exc

    transcript_move = extract_final_move_from_transcript(move_reasoning_transcript)
    validation_error = None
    if mismatch_error is not None:
        validation_error = mismatch_error
    elif chosen_move not in interpreted_moves:
        validation_error = f"Stage 2 selected illegal move {chosen_move}. Interpreted legal moves: {interpreted_moves}"
    elif chosen_move not in moves:
        validation_error = f"Stage 2 selected move {chosen_move}, which is legal on the interpreted board but illegal on the source board legal moves {moves}."
    elif transcript_move is None:
        validation_error = "Stage 2 reasoning_transcript did not end with a valid FINAL_MOVE line."
    elif transcript_move != chosen_move:
        validation_error = f"Stage 2 returned chosen_move {chosen_move}, but reasoning_transcript ended with FINAL_MOVE: {transcript_move}."

    return attach_debug_record(
        AssistantResponse(
            provider=request.provider,
            model=move_model_name if move_model_name == observation_model_name else f"{observation_model_name} -> {move_model_name}",
            observation_model=observation_model_name,
            move_model=move_model_name,
            current_player=request.player,
            legal_moves=moves,
            interpreted_legal_moves=interpreted_moves,
            chosen_move=chosen_move if validation_error is None else None,
            proposed_move=chosen_move,
            interpreted_board=interpreted_board,
            reasoning_transcript=move_reasoning_transcript,
            observation_reasoning_transcript=observation_reasoning_transcript,
            move_reasoning_transcript=move_reasoning_transcript,
            explanation=(
                f"Validated structured move: cell {chosen_move}."
                if validation_error is None
                else "The interpreted board or chosen move failed validation."
            ),
            confidence=None,
            validation_status="valid" if validation_error is None else "invalid",
            validation_error=validation_error,
            prompt_preview=stage2_prompt,
            observation_prompt_preview=stage1_prompt,
            move_prompt_preview=stage2_prompt,
        ),
        run_id=run_id,
        request=request,
        model_name=observation_model_name,
        image_path=debug_image_path,
    )


def build_mock_chat_response(request: ChatRequest) -> ChatResponse:
    latest_user = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
    reply = (
        "Mock chat mode is active. "
        "The real chat path is ready, but this response is coming from the deterministic fallback.\n\n"
        f"Latest user message: {latest_user}\n\n"
        "Try switching the provider to Ollama if you want the same GUI to behave like a real local assistant."
    )
    return ChatResponse(provider=request.provider, model="mock-chat-v1", reply=reply)


def build_ollama_chat_response(request: ChatRequest) -> ChatResponse:
    model_name = request.model or DEFAULT_OLLAMA_MODEL
    prompt_parts = [
        "You are a robotics lab assistant inside a tic-tac-toe demo interface.",
        "Keep answers concise, technical, and directly useful.",
        "Do not output JSON for chat. Respond naturally.",
        "",
        "Conversation:",
    ]
    for message in request.messages:
        prompt_parts.append(f"{message.role.upper()}: {message.content}")
    prompt_parts.append("ASSISTANT:")
    prompt = "\n".join(prompt_parts)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4},
            },
            timeout=(3.0, 90.0),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama chat request failed: {exc}") from exc

    if payload.get("error"):
        raise HTTPException(status_code=502, detail=f"Ollama error: {payload['error']}")

    reply = str(payload.get("response", "")).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Ollama returned an empty chat response.")

    return ChatResponse(provider=request.provider, model=model_name, reply=reply)


camera_service = CameraService(camera_index=int(os.getenv("WEB_TTT_CAMERA_INDEX", "0")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        camera_service.close()


app = FastAPI(title="web_ttt backend", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/providers", response_model=ProviderCatalog)
def provider_catalog() -> ProviderCatalog:
    return list_provider_options()


@app.get("/api/prompt-config", response_model=PromptConfig)
def prompt_config() -> PromptConfig:
    return current_prompt_config()


@app.get("/api/vision/status", response_model=VisionStatus)
def vision_status() -> VisionStatus:
    return camera_service.status()


@app.post("/api/vision/config", response_model=VisionStatus)
def update_vision_config(request: VisionConfigRequest) -> VisionStatus:
    camera_service.set_camera_index(request.camera_index)
    return camera_service.status()


@app.post("/api/assistant/move", response_model=AssistantResponse)
def assistant_move(request: MoveRequest) -> AssistantResponse:
    validate_board(request.board, request.player, analysis_only=request.analysis_only)
    moves = legal_moves(request.board)
    if not moves:
        raise HTTPException(status_code=409, detail="No legal moves remain.")
    if request.provider == "mock":
        return build_mock_response(request, moves)
    return build_ollama_response(request, moves)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.provider == "mock":
        return build_mock_chat_response(request)
    return build_ollama_chat_response(request)


def mjpeg_stream(source: Literal["auto", "synthetic", "camera"]):
    while True:
        if source == "synthetic":
            frame = camera_service._synthetic_frame()
            camera_service._last_source = "synthetic"
        else:
            frame = camera_service.get_frame()
            if source == "camera" and camera_service._last_source != "camera":
                frame = camera_service._synthetic_frame()
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(1 / 15)


@app.get("/vision/stream")
def vision_stream(source: Literal["auto", "synthetic", "camera"] = Query(default="auto")) -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(source), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
