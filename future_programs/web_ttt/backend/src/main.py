from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
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

OLLAMA_URL = os.getenv("WEB_TTT_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("WEB_TTT_OLLAMA_MODEL", "gemma4:e4b")

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


class AssistantResponse(BaseModel):
    provider: str
    model: str
    current_player: Player
    legal_moves: list[int]
    chosen_move: int
    reasoning_transcript: str
    explanation: str
    confidence: float


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
            capture = self._ensure_capture()
            if capture is not None:
                ok, frame = capture.read()
                if ok and frame is not None:
                    self._last_source = "camera"
                    return self._annotate_camera_frame(frame)
                self._release_capture()

            self._last_source = "synthetic"
            return self._synthetic_frame()

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

    def _annotate_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        overlay = annotated.copy()
        cv2.rectangle(overlay, (24, 24), (380, 124), (27, 48, 44), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
        cv2.putText(annotated, "Observed Input", (44, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (244, 240, 232), 2, cv2.LINE_AA)
        cv2.putText(annotated, "Camera preview only. Board logic still uses GUI state.", (44, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 230, 226), 1, cv2.LINE_AA)
        return annotated

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


def validate_board(board: list[Cell], player: Player) -> None:
    if len(board) != 9:
        raise HTTPException(status_code=422, detail="Board must contain exactly 9 cells.")
    if any(cell not in ("", "X", "O") for cell in board):
        raise HTTPException(status_code=422, detail="Board contains invalid cell values.")

    x_count = sum(cell == "X" for cell in board)
    o_count = sum(cell == "O" for cell in board)
    if x_count < o_count or x_count - o_count > 1:
        raise HTTPException(status_code=422, detail="Board state is not reachable in a legal game.")

    winner = board_winner(board)
    if winner is not None:
        raise HTTPException(status_code=409, detail=f"Game is already over. Winner: {winner}.")

    expected = "X" if x_count == o_count else "O"
    if player != expected:
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


def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return response.ok
    except requests.RequestException:
        return False


def list_provider_options() -> ProviderCatalog:
    ollama_is_up = ollama_available()
    return ProviderCatalog(
        default_provider="mock",
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
    chosen_move, rationale = choose_mock_move(request.board, request.player)
    transcript = "\n".join(
        (
            f"Provider: {request.provider}",
            "Mode: GUI-first demo",
            f"Observed board:\n{format_board_for_prompt(request.board)}",
            f"Current player: {request.player}",
            f"Legal moves: {moves}",
            rationale,
            f"Final structured move: {chosen_move}",
        )
    )
    return AssistantResponse(provider=request.provider, model="mock-strategist-v1", current_player=request.player, legal_moves=moves, chosen_move=chosen_move, reasoning_transcript=transcript, explanation=f"{request.player} selects cell {chosen_move}.", confidence=0.68)


def build_ollama_response(request: MoveRequest, moves: list[int]) -> AssistantResponse:
    model_name = request.model or DEFAULT_OLLAMA_MODEL
    schema = {
        "type": "object",
        "properties": {
            "chosen_move": {"type": "integer"},
            "reasoning_transcript": {"type": "string"},
            "explanation": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["chosen_move", "reasoning_transcript", "explanation", "confidence"],
    }
    prompt = "\n".join(
        (
            "You are the tic-tac-toe assistant in a robotics lab demo.",
            f"It is player {request.player}'s turn.",
            "Show a short visible reasoning transcript for the operator.",
            "Your final action must appear only in chosen_move.",
            "Return JSON only.",
            f"Board as rows:\n{format_board_for_prompt(request.board)}",
            f"Raw board list: {request.board}",
            f"Legal moves: {moves}",
            "Required JSON fields:",
            "- chosen_move: integer and one of the legal moves",
            "- reasoning_transcript: short visible thought process for the demo",
            "- explanation: one concise sentence",
            "- confidence: number from 0.0 to 1.0",
        )
    )
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False, "format": schema, "options": {"temperature": 0.2}},
            timeout=(3.0, 90.0),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc
    if payload.get("error"):
        raise HTTPException(status_code=502, detail=f"Ollama error: {payload['error']}")
    try:
        parsed = extract_json_object(str(payload.get("response", "")))
        chosen_move = int(parsed["chosen_move"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama returned invalid structured output: {exc}") from exc
    if chosen_move not in moves:
        raise HTTPException(status_code=502, detail=f"Ollama selected illegal move {chosen_move}. Legal moves: {moves}")
    reasoning_transcript = str(parsed.get("reasoning_transcript", f"I reviewed the legal moves {moves} and selected {chosen_move}.")).strip()
    explanation = str(parsed.get("explanation", f"{request.player} selects cell {chosen_move}.")).strip()
    return AssistantResponse(provider=request.provider, model=model_name, current_player=request.player, legal_moves=moves, chosen_move=chosen_move, reasoning_transcript=reasoning_transcript, explanation=explanation, confidence=clamp_confidence(parsed.get("confidence"), default=0.5))


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


@app.get("/api/vision/status", response_model=VisionStatus)
def vision_status() -> VisionStatus:
    return camera_service.status()


@app.post("/api/vision/config", response_model=VisionStatus)
def update_vision_config(request: VisionConfigRequest) -> VisionStatus:
    camera_service.set_camera_index(request.camera_index)
    return camera_service.status()


@app.post("/api/assistant/move", response_model=AssistantResponse)
def assistant_move(request: MoveRequest) -> AssistantResponse:
    validate_board(request.board, request.player)
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
