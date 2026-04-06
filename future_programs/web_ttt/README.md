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
- assistant endpoint supporting both `mock` and `ollama` providers
- separate chat endpoint supporting both `mock` and `ollama` providers
- move requests can use direct GUI state, a synthetic board image, or the live camera frame as the observation source

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
- `/api/providers`
- `/api/vision/status`
- `/api/assistant/move`
- `/api/chat`
- `/vision/stream`

## Ollama Notes

- The frontend can now request moves through the `ollama` provider.
- Default configuration:
  - URL: `http://127.0.0.1:11434`
  - model: `gemma4:e4b`
- Override with environment variables before starting the backend:

```powershell
$env:WEB_TTT_OLLAMA_URL='http://127.0.0.1:11434'
$env:WEB_TTT_OLLAMA_MODEL='gemma4:e4b'
python main.py
```

- If Ollama is not reachable, the UI still works with the `mock` provider.

## Observation Modes

The move assistant now supports two observation modes:

- `Direct State`: the backend prompt receives the board state directly
- `Live Camera Frame`: the backend captures the current webcam frame and the Ollama move prompt uses that image instead of the raw board state

The backend still validates the chosen move deterministically in both modes.

## Model Comparison

The frontend exposes separate stage 1 and stage 2 Ollama model fields for side-by-side manual comparison.

Current quick presets:

- `gemma4:e4b`
- `gemma4:26b`
- `qwen3.5:9b`

Stage 1 is the observation model.
Stage 2 is the move-reasoning model.

## Dataset Organization

Recommended layout:

```text
debug_images/
  test_prompts/
  datasets/
    paper_mockup_v1/
      records/
      replays/
    real_board_lab_a/
      records/
      replays/
```

Use one dataset folder per camera/board/lighting setup.
This keeps benchmark sets stable instead of mixing all captures into one flat folder.

For live backend captures, set:

```powershell
$env:WEB_TTT_DEBUG_DATASET='paper_mockup_v1'
```

Then start the backend normally. New images and records will be saved into:

```text
debug_images/datasets/paper_mockup_v1/records/
```

## Camera Notes

- If OpenCV can open a camera, the GUI will show a live preview.
- If no camera is available, the backend serves a synthetic fallback stream so the layout still works.
- The current move logic still uses the GUI board state, not camera perception.
- The GUI can now switch camera index dynamically if Windows maps the phone camera to index `0` and the webcam to another index such as `1`.

## Replay Harness

The project includes a replay tool for evaluating models against saved debug images:

- script: `future_programs/web_ttt/tools/replay_debug_images.py`
- input dataset: `debug_images/*.json` plus linked image files
- output folder:
  - flat mode: `debug_images/replays/replay_YYYYMMDD_HHMMSS`
  - dataset mode: `debug_images/datasets/<dataset>/replays/replay_YYYYMMDD_HHMMSS`

Each replay output contains:

- per-record result JSON files
- `summary.json`
- `summary.csv`
- `run_config.json`
- `prompt_used.txt` when a custom prompt file is provided

### Replay Parameters

`replay_debug_images.py` supports:

- `--debug-dir` path to debug records (default is workspace `debug_images`)
- `--dataset` dataset name under `debug_images/datasets/<dataset>/records`
- `--prompt-file` custom prompt template file
- `--model` Ollama model override (otherwise uses model saved in each record)
- `--ollama-url` Ollama base URL
- `--limit` latest N records, `0` means all
- `--record` specific debug record filename (repeatable)
- `--pattern` filename substring filter
- `--timeout-seconds` read timeout per request
- `--retries` retries per record after the first attempt
- `--num-predict` optional max output tokens; omit to match live GUI behavior
- `--temperature` sampling temperature

### Prompt Template Placeholders

When using `--prompt-file`, these placeholders are available:

- `{player}`
- `{source_board_rows}`
- `{source_board_list}`
- `{legal_moves}`
- `{observation_mode}`

### Copy-Paste Replay Examples

Replay latest 8 records with the model stored in each record:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --limit 8 --timeout-seconds 45 --retries 0
```

Replay the latest 8 records from a named dataset:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --dataset paper_mockup_v1 --limit 8 --timeout-seconds 45 --retries 0
```

Replay latest 8 with an explicit model:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --limit 8 --model gemma4:26b --timeout-seconds 45 --retries 0
```

Replay latest 8 with a custom prompt:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --limit 8 --model gemma4:26b --prompt-file C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images\test_prompts\camera_board_analysis_v8.txt --timeout-seconds 45 --retries 0
```

Replay one specific record:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --record web_ttt_20260405_205823_927_camera_frame_analysis.json --model gemma4:26b --timeout-seconds 45 --retries 0
```

Replay records filtered by filename pattern:

```powershell
python future_programs\web_ttt\tools\replay_debug_images.py --pattern 2059 --model gemma4:26b --timeout-seconds 45 --retries 0
```

## Two-Stage Pipeline Replay

The project also includes a two-stage replay tool:

- script: `future_programs/web_ttt/tools/replay_two_stage_pipeline.py`
- stage 1: image -> `interpreted_board`
- stage 2: `interpreted_board` -> legal O move
- output folder:
  - flat mode: `debug_images/replays/pipeline_replay_YYYYMMDD_HHMMSS`
  - dataset mode: `debug_images/datasets/<dataset>/replays/pipeline_replay_YYYYMMDD_HHMMSS`

This is the preferred offline workflow for debugging the full LLM demo chain without touching the live GUI.

Each pipeline replay record stores:

- source image and source board
- stage 1 prompt, interpreted board, transcript, mismatch info, token usage
- stage 2 prompt, chosen move, transcript, legality checks, token usage
- overall pipeline status and total duration

### Two-Stage Prompt Files

Default prompt files:

- stage 1: `debug_images/test_prompts/camera_board_analysis_v8.txt`
- stage 2: `debug_images/test_prompts/move_reasoning_v1.txt`

Stage 2 assumes the assistant always plays `O`.
The pipeline replay is image-based. If the debug folder contains direct-state records with no saved image, they are skipped automatically.
The live web GUI can override both prompts temporarily. `Reset` returns the text areas to these current file-based defaults.

### Two-Stage Copy-Paste Example

Run the full two-stage pipeline on the latest 8 records using `gemma4:26b` for both stages:

```powershell
python future_programs\web_ttt\tools\replay_two_stage_pipeline.py --limit 8 --stage1-model gemma4:26b --stage2-model gemma4:26b --timeout-seconds 45
```

Run the same pipeline on a named dataset:

```powershell
python future_programs\web_ttt\tools\replay_two_stage_pipeline.py --dataset paper_mockup_v1 --limit 8 --stage1-model gemma4:26b --stage2-model gemma4:26b --timeout-seconds 45
```

Run the full two-stage pipeline with different models for the two stages:

```powershell
python future_programs\web_ttt\tools\replay_two_stage_pipeline.py --limit 8 --stage1-model gemma4:26b --stage2-model gemma4:e4b --timeout-seconds 45
```

Run the full two-stage pipeline with a different stage 2 prompt:

```powershell
python future_programs\web_ttt\tools\replay_two_stage_pipeline.py --limit 8 --stage1-model gemma4:26b --stage2-model gemma4:26b --stage2-prompt-file C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images\test_prompts\move_reasoning_v2.txt --timeout-seconds 45
```

Run the pipeline for one specific record:

```powershell
python future_programs\web_ttt\tools\replay_two_stage_pipeline.py --record web_ttt_20260405_205823_927_camera_frame_analysis.json --stage1-model gemma4:26b --stage2-model gemma4:26b --timeout-seconds 45
```

## Replay Comparison

The project also includes a comparison tool for two pipeline replay folders:

- script: `future_programs/web_ttt/tools/compare_pipeline_replays.py`
- input: two `pipeline_replay_*` folders
- output: `comparison.json` and `comparison.csv`

Copy-paste example:

```powershell
python future_programs\web_ttt\tools\compare_pipeline_replays.py --left C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images\replays\pipeline_replay_20260406_181241 --right C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images\replays\pipeline_replay_20260406_183015
```
