# web_ttt Plan

## Recommended Architecture

Use a split architecture:

- frontend: web UI
- backend: Python service
- model provider layer: Ollama and optional Gemini adapters
- rules engine: deterministic tic-tac-toe validation
- execution layer: click driver or robot-safe adapter

## Why This Shape

- The frontend can evolve independently from the robot logic
- Local and cloud models can be swapped without rewriting the app
- Safety-critical logic stays deterministic
- The explanation view becomes a product feature instead of a debug accident

## UI Options

### Option A: Single Portrait Web App With Two Vertical Sections

Layout:

- top board section
- bottom assistant section

Assistant panel content:

- model name
- board interpretation
- visible reasoning transcript
- chosen move
- short explanation
- recent logs

Recommendation:

- Start here

Reason:

- Lowest complexity
- Best operator visibility
- Best fit for a portrait monitor
- Easier than synchronizing multiple windows



## LLM Output Contract

The backend should ask the model for structured JSON like:

```json
{
  "current_player": "O",
  "board": ["X", "", "O", "", "X", "", "", "", ""],
  "legal_moves": [1, 3, 5, 6, 7, 8],
  "reasoning_transcript": "I see X threatening the diagonal, so I should block while keeping my next line open.",
  "chosen_move": 5,
  "explanation": "Blocks X and keeps center pressure.",
  "confidence": 0.82
}
```

The backend must verify:

- the board is valid
- the selected move is legal
- the selected move matches the active player

If validation fails, the move is rejected and logged.

For the demo, the frontend may render `reasoning_transcript` as visible model thinking. The backend must still treat only `chosen_move` as executable.

## Backend Responsibilities

- hold the authoritative board state
- validate all moves
- manage game sessions
- call the selected LLM provider
- stream or return explanation data to the frontend
- support a mode where the model reads the board from the web GUI before camera input exists
- later expose a safe execution hook for Dobot integration

## Frontend Responsibilities

- render the board
- render logs and explanation data
- render a portrait layout suited for a vertical monitor
- let the operator pick provider and model
- let the operator step through moves manually
- show clear error states when the model response is invalid

## Suggested Tech Stack

Conservative path:

- frontend: React + Vite
- backend: FastAPI
- local model calls: Ollama HTTP API

Simpler fallback:

- backend-rendered FastAPI templates

If Gemini is mainly doing frontend drafting, React + Vite is the better target because generated UI code will be easier to adapt there.

## Delivery Phases

### Phase 1

- local standalone web tic-tac-toe
- no robot
- deterministic rules
- portrait UI
- mock visible reasoning data

### Phase 2

- Ollama provider
- Gemma 4 move selection
- visible reasoning panel backed by model output
- separate structured final move validation

### Phase 3

- GUI-first observation mode where the model can inspect the web board
- model comparison harness

### Phase 4

- camera-based board perception
- safe execution integration for click automation or Dobot commands

## First Concrete Build Target

Create a minimal app with:

- browser board
- backend game state
- provider interface
- fake provider returning visible reasoning plus a structured move

That gives a stable shell before adding real models.
