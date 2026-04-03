# Agent Instructions

## Workspace Intent

- `tictactoe/` is a preserved baseline copied from the working Jetson Nano + Dobot setup.
- Treat `tictactoe/` as read-only reference code unless the user explicitly asks to modify it.
- New programs, prototypes, and LLM experiments must be created outside `tictactoe/`.

## Default Working Rules

- Put all new work under `future_programs/<project_name>/`.
- Keep each new program self-contained with its own `README.md`.
- Add explicit dependency files for new programs such as `requirements.txt` or `pyproject.toml`.
- When reusing ideas from `tictactoe/`, copy relevant logic into the new folder instead of editing the original.
- Do not rewrite calibration files, assets, or Dobot control code inside `tictactoe/` unless asked.

## Environment Assumptions

- The active workspace may be on Windows, but the original robot app runs on Jetson Nano / Linux.
- For robot-facing code, prefer Linux-friendly paths, serial-port configurability, and deployment notes.
- Avoid assuming the current machine has Dobot hardware, Ollama, or a GUI available unless verified.

## External Model Tooling

- Gemini CLI may be used in a separate terminal for development support tasks.
- Treat Gemini CLI as a sidecar tool for ideation, drafting, UI exploration, and alternative implementations.
- Codex remains the primary agent for making deliberate repo changes in this workspace.
- If Gemini output is brought into the repo, integrate it intentionally rather than trusting it as final code.

## Parallel Agent Workflow

- Do not let Codex and Gemini edit the same file at the same time.
- Prefer a handoff model:
  - Gemini proposes or drafts
  - Codex integrates, refines, and verifies
- Put Gemini-generated experiments in `future_programs/<project_name>/` unless the user explicitly wants another location.
- When using Gemini for frontend work, keep the output isolated until it has been reviewed and adapted to the repo.
- For robotics, safety-critical logic, and motion planning, prefer deterministic Python code over LLM-generated direct control.

## Collaboration Expectations

- State clearly which folders will be changed before editing files.
- Keep summaries explicit about whether `tictactoe/` was untouched.
- Prefer small, isolated prototypes over invasive repo-wide changes.
