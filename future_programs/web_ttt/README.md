# web_ttt

Planned web-based successor to the Tkinter tic-tac-toe system.

## Goal

Replace the current desktop-oriented control flow with a browser-based interface that can:

- show the tic-tac-toe board state
- show model move suggestions and explanations
- switch between local and cloud LLM providers
- later connect to safe execution paths for the Dobot or other controllers

## Non-Goal

This project does not modify the original `tictactoe/` reference implementation.

## Initial Product Shape

Recommended first version:

- one web page
- portrait-oriented layout for a vertical monitor
- top: tic-tac-toe board and game controls
- bottom: assistant panel with model output, visible reasoning, final move, and logs
- camera preview card in the top section beside the board

This is simpler than managing separate windows and matches the intended robotics lab display.

## Model Strategy

Default model path:

- local Gemma 4 through Ollama

Optional model paths:

- Gemini for comparison
- other Ollama models for benchmarking

## Explanation Strategy

For the lab demo, the UI should visibly show the model thinking through the move.

Implementation rule:

- show a visible reasoning or transcript panel for demonstration purposes
- keep the final executable move in a separate structured field

The model response should therefore contain fields such as:

- current player
- detected board state
- candidate moves
- selected move
- reasoning transcript
- short explanation
- confidence

This gives a visible LLM-driven experience while still allowing deterministic validation of the final move.

## Perception Strategy

Initial target:

- use the web GUI itself as the board being observed
- allow a side camera preview in the GUI even before camera-based board parsing exists

Later target:

- replace GUI observation with a camera looking at the physical board

This preserves the same high-level pipeline while reducing complexity in the first implementation.

## Current Prototype

The current implementation now has a real split stack:

- `backend/src/main.py`: FastAPI backend
- `frontend/`: React + Vite frontend
- portrait-oriented board and assistant layout
- synthetic or live camera preview served by the backend
- mock assistant endpoint returning visible reasoning plus a structured move

The original static prototype is still present as a fallback reference:

- `serve.py`
- `web/`

## Running The Current App

Backend:

```bash
cd future_programs/web_ttt/backend/src
python main.py
```

Frontend:

```bash
cd future_programs/web_ttt/frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Current Backend Endpoints

- `/api/health`
- `/api/vision/status`
- `/api/assistant/move`
- `/vision/stream`

## Camera Notes

- If OpenCV can open a camera, the GUI will show a live preview.
- If no camera is available, the backend serves a synthetic fallback stream so the layout still works.
- The current move logic still uses the GUI board state, not camera perception.
