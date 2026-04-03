# TODO

## Current Direction

Build the next generation of the Dobot tic-tac-toe system outside `tictactoe/`, with a web-based UI and a pluggable LLM backend. The baseline target is local Gemma 4 through Ollama, with room for cloud providers such as Gemini when useful.

## Near Term

- Define the architecture for the web-based replacement
- Decide on frontend stack and backend stack
- Create a clean project under `future_programs/`
- Separate board state, rules, model provider, and execution layers
- Design a portrait-oriented assistant panel for visible model reasoning and final move output

## Important Constraints

- Keep `tictactoe/` unchanged as the working reference system
- Do not let the LLM directly control robot coordinates or low-level motion
- Keep tic-tac-toe legality checks deterministic
- Prefer local inference first, with cloud models as optional adapters

## Recommended Build Order

1. Web UI prototype for a portrait monitor with board on top and assistant panel on bottom
2. Backend API with deterministic tic-tac-toe rules
3. Ollama provider for local Gemma 4
4. Visible reasoning transcript plus separate structured final move output
5. GUI-first board-state ingestion so the web app can act as the initial perception target
6. Safe executor layer for UI clicking or robot commands
7. Camera-based board perception
8. Optional Gemini adapter for comparison and frontend assistance

## Open Decisions

- Exact visual treatment of the portrait layout
- React-based frontend or simpler server-rendered UI
- Polling versus streaming updates between backend and frontend
- Whether the first version should read direct board state, screenshot the web GUI, or support both modes
