import { useEffect, useState } from 'react'
import './App.css'

type Player = 'X' | 'O'
type Cell = '' | Player

type AssistantResponse = {
  provider: string
  model: string
  observation_model: string | null
  move_model: string | null
  debug_dataset: string | null
  current_player: Player
  legal_moves: number[]
  interpreted_legal_moves: number[] | null
  chosen_move: number | null
  proposed_move: number | null
  interpreted_board: Cell[] | null
  reasoning_transcript: string
  observation_reasoning_transcript: string | null
  move_reasoning_transcript: string | null
  explanation: string
  confidence: number | null
  validation_status: 'valid' | 'invalid'
  validation_error: string | null
  prompt_preview: string | null
  observation_prompt_preview: string | null
  move_prompt_preview: string | null
  debug_image_path: string | null
  debug_record_path: string | null
}

type VisionStatus = {
  active_source: 'camera' | 'synthetic'
  camera_index: number
  streaming: boolean
  note: string
}

type ProviderOption = {
  id: 'mock' | 'ollama'
  label: string
  available: boolean
  default_model: string | null
  note: string
}

type ProviderCatalog = {
  default_provider: 'mock' | 'ollama'
  providers: ProviderOption[]
}

type PromptConfig = {
  stage1_prompt: string
  stage2_prompt: string
  stage1_prompt_file: string | null
  stage2_prompt_file: string | null
}

type ObservationMode = 'direct_state' | 'rendered_image' | 'camera_frame'
type GameMode = 'human_vs_human' | 'human_vs_assistant'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

const modelOptions = ['gemma4:e4b', 'gemma4:26b', 'qwen3.5:9b', 'gemini-3-flash-preview:cloud']

const winningLines = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
]

function findWinner(board: Cell[]): Player | null {
  for (const [a, b, c] of winningLines) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a]
    }
  }
  return null
}

function legalMoves(board: Cell[]) {
  return board.flatMap((cell, index) => (cell === '' ? [index] : []))
}

function formatBoard(board: Cell[]) {
  return Array.from({ length: 3 }, (_, row) =>
    board
      .slice(row * 3, row * 3 + 3)
      .map((cell) => (cell === '' ? '.' : cell))
      .join(' '),
  ).join('\n')
}

function observationModeLabel(mode: ObservationMode) {
  if (mode === 'rendered_image') {
    return 'Synthetic board image'
  }
  if (mode === 'camera_frame') {
    return 'Live camera frame'
  }
  return 'Direct GUI state'
}

function describeAssistantModels(response: AssistantResponse) {
  if (response.move_model) {
    if (response.observation_model && response.observation_model !== response.move_model) {
      return `${response.provider}:${response.observation_model} -> ${response.move_model}`
    }
    return `${response.provider}:${response.move_model}`
  }
  if (response.observation_model) {
    return `${response.provider}:${response.observation_model}`
  }
  return `${response.provider}:${response.model}`
}

function nextPlayer(board: Cell[]): Player {
  const xCount = board.filter((cell) => cell === 'X').length
  const oCount = board.filter((cell) => cell === 'O').length
  return xCount === oCount ? 'X' : 'O'
}

function createBoardSnapshot(board: Cell[]): string {
  const canvas = document.createElement('canvas')
  const size = 720
  const padding = 36
  const gap = 18
  const cellSize = (size - padding * 2 - gap * 2) / 3
  canvas.width = size
  canvas.height = size

  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Could not create board snapshot canvas.')
  }

  const gradient = context.createLinearGradient(0, 0, size, size)
  gradient.addColorStop(0, '#f8f2e8')
  gradient.addColorStop(1, '#e7dcc8')
  context.fillStyle = gradient
  context.fillRect(0, 0, size, size)

  board.forEach((cell, index) => {
    const row = Math.floor(index / 3)
    const col = index % 3
    const x = padding + col * (cellSize + gap)
    const y = padding + row * (cellSize + gap)

    context.fillStyle = '#fbf7ef'
    context.strokeStyle = 'rgba(138, 116, 83, 0.16)'
    context.lineWidth = 4
    context.beginPath()
    context.roundRect(x, y, cellSize, cellSize, 28)
    context.fill()
    context.stroke()

    if (cell === 'X') {
      context.strokeStyle = '#c84c35'
      context.lineWidth = 16
      context.lineCap = 'round'
      context.beginPath()
      context.moveTo(x + 46, y + 46)
      context.lineTo(x + cellSize - 46, y + cellSize - 46)
      context.moveTo(x + cellSize - 46, y + 46)
      context.lineTo(x + 46, y + cellSize - 46)
      context.stroke()
    }

    if (cell === 'O') {
      context.strokeStyle = '#2d7077'
      context.lineWidth = 16
      context.beginPath()
      context.arc(x + cellSize / 2, y + cellSize / 2, cellSize / 2 - 44, 0, Math.PI * 2)
      context.stroke()
    }
  })

  return canvas.toDataURL('image/png').split(',')[1]
}

function App() {
  const [board, setBoard] = useState<Cell[]>(Array(9).fill(''))
  const [assistantSourceBoard, setAssistantSourceBoard] = useState<Cell[] | null>(null)
  const [currentPlayer, setCurrentPlayer] = useState<Player>('X')
  const [gameMode, setGameMode] = useState<GameMode>('human_vs_human')
  const [winner, setWinner] = useState<Player | null>(null)
  const [assistant, setAssistant] = useState<AssistantResponse | null>(null)
  const [vision, setVision] = useState<VisionStatus | null>(null)
  const [providerCatalog, setProviderCatalog] = useState<ProviderCatalog | null>(null)
  const [promptConfig, setPromptConfig] = useState<PromptConfig | null>(null)
  const [selectedProvider, setSelectedProvider] = useState<'mock' | 'ollama'>('ollama')
  const [stage1Model, setStage1Model] = useState('gemma4:26b')
  const [stage2Model, setStage2Model] = useState('gemma4:26b')
  const [stage1PromptDraft, setStage1PromptDraft] = useState('')
  const [stage2PromptDraft, setStage2PromptDraft] = useState('')
  const [observationMode, setObservationMode] = useState<ObservationMode>('camera_frame')
  const [streamNonce, setStreamNonce] = useState(0)
  const [assistantTab, setAssistantTab] = useState<'reasoning' | 'chat' | 'logs'>('reasoning')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Chat ready. Ask about the game, the camera feed, or the robotics demo setup.',
    },
  ])
  const [chatDraft, setChatDraft] = useState('')
  const [sendingChat, setSendingChat] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [requestingMove, setRequestingMove] = useState(false)
  const [logs, setLogs] = useState<string[]>([
    'Prototype ready. Set up a board state, then analyze the board or run the full two-stage assistant.',
  ])

  useEffect(() => {
    let cancelled = false

    const loadVisionStatus = async () => {
      try {
        const response = await fetch('/api/vision/status')
        if (!response.ok) {
          throw new Error(`Vision status failed with ${response.status}`)
        }
        const data = (await response.json()) as VisionStatus
        if (!cancelled) {
          setVision(data)
        }
      } catch (fetchError) {
        if (!cancelled) {
          setVision(null)
          setError((fetchError as Error).message)
        }
      }
    }

    loadVisionStatus()
    const interval = window.setInterval(loadVisionStatus, 4000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const loadPromptConfig = async () => {
      try {
        const response = await fetch('/api/prompt-config')
        if (!response.ok) {
          throw new Error(`Prompt config failed with ${response.status}`)
        }
        const data = (await response.json()) as PromptConfig
        if (!cancelled) {
          setPromptConfig(data)
          setStage1PromptDraft(data.stage1_prompt)
          setStage2PromptDraft(data.stage2_prompt)
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError((fetchError as Error).message)
        }
      }
    }

    loadPromptConfig()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const loadProviders = async () => {
      try {
        const response = await fetch('/api/providers')
        if (!response.ok) {
          throw new Error(`Provider catalog failed with ${response.status}`)
        }
        const data = (await response.json()) as ProviderCatalog
        if (!cancelled) {
          setProviderCatalog(data)
          setSelectedProvider(data.default_provider)
          const ollama = data.providers.find((provider) => provider.id === 'ollama')
          if (ollama?.default_model) {
            setStage1Model(ollama.default_model)
            setStage2Model(ollama.default_model)
          }
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError((fetchError as Error).message)
        }
      }
    }

    loadProviders()

    return () => {
      cancelled = true
    }
  }, [])

  const appendLog = (message: string) => {
    setLogs((current) => [`${new Date().toLocaleTimeString()}  ${message}`, ...current].slice(0, 8))
  }

  const refreshCameraStream = () => {
    setStreamNonce((value) => value + 1)
  }

  const resetGame = () => {
    setBoard(Array(9).fill(''))
    setAssistantSourceBoard(null)
    setCurrentPlayer('X')
    setWinner(null)
    setAssistant(null)
    setError(null)
    appendLog('New game started.')
  }

  const applyBoardState = (nextBoard: Cell[]) => {
    const nextWinner = findWinner(nextBoard)
    setBoard(nextBoard)
    setWinner(nextWinner)
    if (nextWinner) {
      setCurrentPlayer(nextWinner)
      return
    }
    setCurrentPlayer(legalMoves(nextBoard).length === 0 ? 'X' : nextPlayer(nextBoard))
  }

  const handleCellClick = (index: number) => {
    if (winner || board[index] !== '') {
      return
    }

    if (gameMode === 'human_vs_assistant' && currentPlayer !== 'X') {
      return
    }

    const nextBoard = board.slice()
    nextBoard[index] = currentPlayer
    applyBoardState(nextBoard)
    appendLog(`Placed ${currentPlayer} at cell ${index}.`)
    setAssistantSourceBoard(null)
    setAssistant(null)
    setError(null)
  }

  const requestAssistant = async (analysisOnly: boolean) => {
    if (selectedProvider === 'ollama') {
      if (!stage1Model.trim()) {
        setError('Choose a stage 1 Ollama model tag before requesting the assistant.')
        appendLog('Assistant request rejected because the stage 1 model field is empty.')
        return
      }
      if (!analysisOnly && !stage2Model.trim()) {
        setError('Choose a stage 2 Ollama model tag before requesting the assistant.')
        appendLog('Assistant request rejected because the stage 2 model field is empty.')
        return
      }
    }

    setRequestingMove(true)
    setError(null)

    try {
      const sourceBoard = board.slice()
      setAssistantSourceBoard(sourceBoard)
      let boardImageBase64: string | undefined
      if (observationMode === 'rendered_image') {
        boardImageBase64 = createBoardSnapshot(sourceBoard)
      }
      const response = await fetch('/api/assistant/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          board: sourceBoard,
          player: currentPlayer,
          provider: selectedProvider,
          stage1_model: selectedProvider === 'ollama' ? stage1Model.trim() : undefined,
          stage2_model: selectedProvider === 'ollama' ? stage2Model.trim() : undefined,
          stage1_prompt_override: selectedProvider === 'ollama' ? stage1PromptDraft : undefined,
          stage2_prompt_override: selectedProvider === 'ollama' ? stage2PromptDraft : undefined,
          observation_mode: observationMode,
          board_image_base64: boardImageBase64,
          analysis_only: analysisOnly,
        }),
      })

      const payload = (await response.json()) as AssistantResponse | { detail?: string }
      if (!response.ok) {
        throw new Error(
          'detail' in payload && payload.detail
            ? payload.detail
            : `Assistant request failed with ${response.status}`,
        )
      }

      const data = payload as AssistantResponse
      setAssistant(data)
      if (data.validation_status === 'invalid') {
        setError(data.validation_error ?? 'The board analysis failed validation.')
        appendLog(`Assistant returned invalid output using ${describeAssistantModels(data)}.`)
        return
      }
      if (!analysisOnly && data.chosen_move !== null) {
        const nextBoard = sourceBoard.slice()
        nextBoard[data.chosen_move] = 'O'
        applyBoardState(nextBoard)
        appendLog(`Assistant played O at cell ${data.chosen_move} using ${describeAssistantModels(data)}.`)
      } else {
        appendLog(`Board analyzed using ${describeAssistantModels(data)} in ${observationMode}.`)
      }
    } catch (requestError) {
      const message = (requestError as Error).message
      setError(message)
      appendLog(`Assistant request failed: ${message}`)
    } finally {
      setRequestingMove(false)
    }
  }

  const handleAnalyzeBoard = async () => {
    await requestAssistant(true)
  }

  const handleRunAssistantMove = async () => {
    await requestAssistant(false)
  }

  const handleSendChat = async () => {
    const trimmed = chatDraft.trim()
    if (!trimmed) {
      return
    }
    if (selectedProvider === 'ollama' && !stage2Model.trim()) {
      setError('Choose an Ollama model tag before sending chat.')
      appendLog('Chat request rejected because the Ollama model field is empty.')
      return
    }

    const nextMessages = [...chatMessages, { role: 'user' as const, content: trimmed }]
    setChatMessages(nextMessages)
    setChatDraft('')
    setSendingChat(true)
    setError(null)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: selectedProvider,
          model: selectedProvider === 'ollama' ? stage2Model.trim() : undefined,
          messages: nextMessages,
        }),
      })

      const payload = (await response.json()) as { reply?: string; detail?: string; model?: string }
      if (!response.ok || !payload.reply) {
        throw new Error(
          payload.detail ? payload.detail : `Chat request failed with ${response.status}`,
        )
      }

      setChatMessages((current) => [...current, { role: 'assistant', content: payload.reply! }])
      appendLog(`Chat reply received from ${selectedProvider}${payload.model ? `:${payload.model}` : ''}.`)
      setAssistantTab('chat')
    } catch (requestError) {
      const message = (requestError as Error).message
      setError(message)
      appendLog(`Chat request failed: ${message}`)
    } finally {
      setSendingChat(false)
    }
  }

  const updateCameraIndex = async (cameraIndex: number) => {
    if (cameraIndex < 0) {
      return
    }

    try {
      const response = await fetch('/api/vision/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera_index: cameraIndex }),
      })
      const payload = (await response.json()) as VisionStatus | { detail?: string }
      if (!response.ok) {
        throw new Error(
          'detail' in payload && payload.detail
            ? payload.detail
            : `Camera update failed with ${response.status}`,
        )
      }

      setVision(payload as VisionStatus)
      refreshCameraStream()
      appendLog(`Switched camera index to ${cameraIndex}.`)
      setError(null)
    } catch (requestError) {
      const message = (requestError as Error).message
      setError(message)
      appendLog(`Camera switch failed: ${message}`)
    }
  }

  const boardFull = legalMoves(board).length === 0
  const canRunAssistantMove =
    gameMode === 'human_vs_assistant' &&
    currentPlayer === 'O' &&
    !winner &&
    !boardFull &&
    !requestingMove
  const statusText = winner
    ? `Winner: ${winner}`
    : boardFull
      ? 'Draw'
      : `Turn: ${currentPlayer}`

  return (
    <main className="app-shell">
      <section className="hero-bar">
        <div>
          <p className="eyebrow">Robot Tic-Tac-Toe</p>
          <h1>Vision-Guided Decision Pipeline</h1>
        </div>
        <div className="hero-pills">
          <span className="pill">{statusText}</span>
          <span className="pill">Observation: {observationModeLabel(observationMode)}</span>
          <span className="pill">Camera: {vision?.active_source === 'camera' ? 'live' : 'synthetic'}</span>
        </div>
      </section>

      <section className="top-zone">
        <article className="panel board-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Interactive Board</p>
              <h2>Game State</h2>
            </div>
            <div className="button-row">
              <button type="button" onClick={resetGame}>
                New Game
              </button>
              <button type="button" onClick={handleAnalyzeBoard} disabled={requestingMove}>
                {requestingMove ? 'Working...' : 'Analyze Board'}
              </button>
              <button type="button" onClick={handleRunAssistantMove} disabled={!canRunAssistantMove}>
                {requestingMove ? 'Working...' : 'Run Assistant Move'}
              </button>
            </div>
          </div>

          <div className="status-strip">
            <div className="status-card board-status-card">
              <span className="status-label">Operator Mode</span>
              <select
                value={gameMode}
                onChange={(event) => {
                  setGameMode(event.target.value as GameMode)
                  resetGame()
                }}
              >
                <option value="human_vs_human">Human vs Human</option>
                <option value="human_vs_assistant">Human X vs Assistant O</option>
              </select>
            </div>
            <div className="status-card board-status-card">
              <span className="status-label">Observed board</span>
              <strong>{observationModeLabel(observationMode)}</strong>
            </div>
          </div>

          <div className="board-grid">
            {board.map((cell, index) => (
              <button
                type="button"
                key={index}
                className={`board-cell ${cell === 'X' ? 'mark-x' : ''} ${cell === 'O' ? 'mark-o' : ''}`.trim()}
                onClick={() => handleCellClick(index)}
                aria-label={`Cell ${index}`}
              >
                {cell}
              </button>
            ))}
          </div>
        </article>

        <article className="panel camera-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Observed Input</p>
              <h2>Camera Stream</h2>
            </div>
            <div className="camera-badge">
              {vision?.active_source === 'camera' ? 'Live camera' : 'Synthetic preview'}
            </div>
          </div>

          <div className="camera-frame">
            <div className="camera-hud-brackets" />
            <div className="camera-hud-bottom" />
            <div className="camera-scanline" />
            <div className="camera-live-badge">
              <div className="badge-dot" />
              <span className="badge-text">Vision System Active</span>
            </div>
            <img src={`/vision/stream?source=auto&v=${streamNonce}`} alt="Observed input stream" />
          </div>

          <div className="status-strip">
            <div className="status-card camera-status-card">
              <span className="status-label">Current source</span>
              <strong>{vision?.active_source ?? 'unknown'}</strong>
            </div>
            <div className="status-card camera-switch-card camera-status-card">
              <span className="status-label">Camera index</span>
              <div className="camera-switch-controls">
                <button type="button" onClick={() => updateCameraIndex((vision?.camera_index ?? 0) - 1)}>
                  -
                </button>
                <strong>{vision?.camera_index ?? 0}</strong>
                <button type="button" onClick={() => updateCameraIndex((vision?.camera_index ?? 0) + 1)}>
                  +
                </button>
              </div>
            </div>
          </div>
        </article>
      </section>

      <section className="panel assistant-panel">
        <div className="panel-header assistant-header">
          <div>
            <p className="panel-kicker">Assistant</p>
            <h2>Two-Stage AI Reasoning</h2>
          </div>
          <div className="provider-controls">
            <label className="provider-field">
              <span className="status-label">Provider</span>
              <select
                value={selectedProvider}
                onChange={(event) => setSelectedProvider(event.target.value as 'mock' | 'ollama')}
              >
                {(providerCatalog?.providers ?? []).map((provider) => (
                  <option
                    key={provider.id}
                    value={provider.id}
                    disabled={!provider.available && provider.id !== 'mock'}
                  >
                    {provider.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="provider-field">
              <span className="status-label">Stage 1 Model</span>
              <select
                value={stage1Model}
                onChange={(event) => setStage1Model(event.target.value)}
                disabled={selectedProvider !== 'ollama'}
              >
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <small className="field-note">Observation model for board interpretation.</small>
            </label>
            <label className="provider-field">
              <span className="status-label">Stage 2 Model</span>
              <select
                value={stage2Model}
                onChange={(event) => setStage2Model(event.target.value)}
                disabled={selectedProvider !== 'ollama'}
              >
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <small className="field-note">Move model for legal O-move reasoning. Chat also uses this model.</small>
            </label>
            <label className="provider-field">
              <span className="status-label">Observation Mode</span>
              <select
                value={observationMode}
                onChange={(event) => setObservationMode(event.target.value as ObservationMode)}
              >
                <option value="direct_state">Direct State</option>
                <option value="rendered_image">Synthetic Board Image</option>
                <option value="camera_frame">Live Camera Frame</option>
              </select>
            </label>
          </div>
        </div>

        <details className="prompt-editor">
          <summary>Prompt Overrides</summary>
          <div className="prompt-editor-grid">
            <article className="prompt-editor-card">
              <div className="prompt-editor-header">
                <div>
                  <span className="status-label">Stage 1 Prompt</span>
                  <strong>Observation Prompt Override</strong>
                </div>
                <button
                  type="button"
                  onClick={() => setStage1PromptDraft(promptConfig?.stage1_prompt ?? '')}
                >
                  Reset
                </button>
              </div>
              <small className="field-note">
                Current file: {promptConfig?.stage1_prompt_file ?? 'n/a'}
              </small>
              <textarea
                value={stage1PromptDraft}
                onChange={(event) => setStage1PromptDraft(event.target.value)}
                spellCheck={false}
              />
            </article>

            <article className="prompt-editor-card">
              <div className="prompt-editor-header">
                <div>
                  <span className="status-label">Stage 2 Prompt</span>
                  <strong>Move Prompt Override</strong>
                </div>
                <button
                  type="button"
                  onClick={() => setStage2PromptDraft(promptConfig?.stage2_prompt ?? '')}
                >
                  Reset
                </button>
              </div>
              <small className="field-note">
                Current file: {promptConfig?.stage2_prompt_file ?? 'n/a'}
              </small>
              <textarea
                value={stage2PromptDraft}
                onChange={(event) => setStage2PromptDraft(event.target.value)}
                spellCheck={false}
              />
            </article>
          </div>
        </details>

        <div className="assistant-tabs">
          <button
            type="button"
            className={`tab-button ${assistantTab === 'reasoning' ? 'tab-active' : ''}`.trim()}
            onClick={() => setAssistantTab('reasoning')}
          >
            Reasoning
          </button>
          <button
            type="button"
            className={`tab-button ${assistantTab === 'chat' ? 'tab-active' : ''}`.trim()}
            onClick={() => setAssistantTab('chat')}
          >
            Chat
          </button>
          <button
            type="button"
            className={`tab-button ${assistantTab === 'logs' ? 'tab-active' : ''}`.trim()}
            onClick={() => setAssistantTab('logs')}
          >
            Logs
          </button>
        </div>

        {assistantTab === 'reasoning' ? (
          <div className="assistant-grid">
            <div className="reasoning-top-row">
              <article className="assistant-card compact-card">
                <h3>Source Board</h3>
                <pre className="mono-panel compact-panel">
                  {assistantSourceBoard
                    ? formatBoard(assistantSourceBoard)
                    : 'No analysis request yet.'}
                </pre>
              </article>

              <article className="assistant-card compact-card">
                <h3>Interpreted Board</h3>
                <pre className="mono-panel compact-panel">
                  {assistant?.interpreted_board
                    ? formatBoard(assistant.interpreted_board)
                    : 'No interpreted board yet.'}
                </pre>
              </article>

              <article className="assistant-card compact-card">
                <h3>Move Output</h3>
                <div className="move-card">
                  <span className={`move-index ${assistant?.validation_status === 'invalid' ? 'move-index-invalid' : ''}`.trim()}>
                    {assistant?.chosen_move ?? assistant?.proposed_move ?? '-'}
                  </span>
                  <p>{assistant?.explanation ?? 'No analysis requested yet.'}</p>
                  {assistant?.interpreted_legal_moves?.length ? (
                    <small>Interpreted legal moves: {assistant.interpreted_legal_moves.join(', ')}</small>
                  ) : null}
                  {assistant?.validation_status === 'invalid' ? (
                    <small className="validation-warning">
                      Rejected: {assistant.validation_error}
                    </small>
                  ) : null}
                </div>
              </article>
            </div>

            <div className="transcript-row">
              <article className="assistant-card transcript-card">
                <h3>Observation Reasoning</h3>
                <small className="status-label">
                  Model: {assistant?.observation_model ?? (selectedProvider === 'ollama' ? stage1Model : 'mock-strategist-v1')}
                </small>
                <pre className="mono-panel transcript-panel">
                  {assistant?.observation_reasoning_transcript ??
                    assistant?.reasoning_transcript ??
                    'No observation reasoning yet.'}
                </pre>
              </article>

              <article className="assistant-card transcript-card">
                <h3>Move Reasoning</h3>
                <small className="status-label">
                  Model: {assistant?.move_model ?? (selectedProvider === 'ollama' ? stage2Model : 'mock-strategist-v1')}
                </small>
                <pre className="mono-panel transcript-panel">
                  {assistant?.move_reasoning_transcript ??
                    (assistant?.chosen_move != null ? assistant.reasoning_transcript : 'No move reasoning yet.')}
                </pre>
              </article>
            </div>

            <article className="assistant-card transcript-card">
              <h3>Observation Prompt</h3>
              <pre className="mono-panel transcript-panel">
                {assistant?.observation_prompt_preview ??
                  assistant?.prompt_preview ??
                  'No observation prompt yet.\n\nRun analysis to inspect the exact stage 1 prompt sent to the model.'}
              </pre>
            </article>

            <article className="assistant-card transcript-card">
              <h3>Move Prompt</h3>
              <pre className="mono-panel transcript-panel">
                {assistant?.move_prompt_preview ??
                  'No move prompt yet.\n\nRun Assistant Move to inspect the exact stage 2 prompt sent to the model.'}
              </pre>
            </article>
          </div>
        ) : null}

        {assistantTab === 'chat' ? (
          <div className="chat-panel">
            <div className="chat-thread">
              {chatMessages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`chat-bubble ${message.role === 'assistant' ? 'chat-assistant' : 'chat-user'}`.trim()}
                >
                  <span className="status-label">{message.role}</span>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>
            <div className="chat-composer">
              <textarea
                value={chatDraft}
                onChange={(event) => setChatDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    void handleSendChat()
                  }
                }}
                placeholder="Ask the assistant about the board, the camera, or the demo setup."
                rows={4}
              />
              <div className="chat-actions">
                <button type="button" onClick={() => setChatMessages([{ role: 'assistant', content: 'Chat reset. Ask a new question.' }])}>
                  Clear Chat
                </button>
                <button type="button" onClick={handleSendChat} disabled={sendingChat}>
                  {sendingChat ? 'Sending...' : 'Send Chat'}
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {assistantTab === 'logs' ? (
          <article className="assistant-card">
            <h3>Event Log</h3>
            <ul className="log-list">
              {logs.map((entry, index) => (
                <li key={`${entry}-${index}`}>{entry}</li>
              ))}
            </ul>
          </article>
        ) : null}

        {error ? <p className="error-banner">{error}</p> : null}
      </section>
    </main>
  )
}

export default App
