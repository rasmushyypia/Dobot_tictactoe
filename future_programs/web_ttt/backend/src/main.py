from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


Player = Literal["X", "O"]
Cell = Literal["", "X", "O"]


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
    provider: str = "mock"


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

    def _annotate_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        overlay = annotated.copy()
        cv2.rectangle(overlay, (24, 24), (380, 124), (27, 48, 44), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
        cv2.putText(
            annotated,
            "Observed Input",
            (44, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (244, 240, 232),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Camera preview only. Board logic still uses GUI state.",
            (44, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 230, 226),
            1,
            cv2.LINE_AA,
        )
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

        cv2.putText(
            frame,
            "web_ttt vision preview",
            (36, 64),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (246, 238, 228),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "No live camera detected. Synthetic board feed active.",
            (36, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (213, 223, 219),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            time.strftime("Generated %H:%M:%S"),
            (36, self.height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (213, 223, 219),
            2,
            cv2.LINE_AA,
        )
        return frame


def board_winner(board: list[Cell]) -> Cell | None:
    for a, b, c in WINNING_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def legal_moves(board: list[Cell]) -> list[int]:
    return [index for index, cell in enumerate(board) if cell == ""]


def validate_board(board: list[Cell], player: Player) -> tuple[int, int]:
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
        raise HTTPException(
            status_code=422,
            detail=f"Expected player {expected} based on board counts, but received {player}.",
        )

    return x_count, o_count


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


camera_service = CameraService(camera_index=int(os.getenv("WEB_TTT_CAMERA_INDEX", "0")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        camera_service.close()


app = FastAPI(title="web_ttt backend", version="0.1.0", lifespan=lifespan)
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


@app.get("/api/vision/status", response_model=VisionStatus)
def vision_status() -> VisionStatus:
    return camera_service.status()


@app.post("/api/assistant/move", response_model=AssistantResponse)
def assistant_move(request: MoveRequest) -> AssistantResponse:
    validate_board(request.board, request.player)
    moves = legal_moves(request.board)
    if not moves:
        raise HTTPException(status_code=409, detail="No legal moves remain.")

    chosen_move, rationale = choose_mock_move(request.board, request.player)
    transcript = "\n".join(
        (
            f"Provider: {request.provider}",
            "Mode: GUI-first demo",
            f"Observed board: {request.board}",
            f"Current player: {request.player}",
            f"Legal moves: {moves}",
            rationale,
            f"Final structured move: {chosen_move}",
        )
    )

    return AssistantResponse(
        provider=request.provider,
        model="mock-strategist-v1",
        current_player=request.player,
        legal_moves=moves,
        chosen_move=chosen_move,
        reasoning_transcript=transcript,
        explanation=f"{request.player} selects cell {chosen_move}.",
        confidence=0.68,
    )


def mjpeg_stream(source: Literal["auto", "synthetic", "camera"]) -> bytes:
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
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        time.sleep(1 / 15)


@app.get("/vision/stream")
def vision_stream(source: Literal["auto", "synthetic", "camera"] = Query(default="auto")) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
