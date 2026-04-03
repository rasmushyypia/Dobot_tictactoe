import { useEffect, useState } from 'react'
import './App.css'

type Player = 'X' | 'O'
type Cell = '' | Player

type AssistantResponse = {
  provider: string
  model: string
  current_player: Player
  legal_moves: number[]
  chosen_move: number
  reasoning_transcript: string
  explanation: string
  confidence: number
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

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

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

function nextPlayer(board: Cell[]): Player {
  const xCount = board.filter((cell) => cell === 'X').length
  const oCount = board.filter((cell) => cell === 'O').length
  return xCount === oCount ? 'X' : 'O'
}

function App() {
  const [board, setBoard] = useState<Cell[]>(Array(9).fill(''))
  const [currentPlayer, setCurrentPlayer] = useState<Player>('X')
  const [winner, setWinner] = useState<Player | null>(null)
  const [assistant, setAssistant] = useState<AssistantResponse | null>(null)
  const [vision, setVision] = useState<VisionStatus | null>(null)
  const [providerCatalog, setProviderCatalog] = useState<ProviderCatalog | null>(null)
  const [selectedProvider, setSelectedProvider] = useState<'mock' | 'ollama'>('mock')
  const [ollamaModel, setOllamaModel] = useState('gemma4:e4b')
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
    'Prototype ready. Human places X on the board. Assistant responds as O.',
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
            setOllamaModel(ollama.default_model)
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
    if (winner || currentPlayer !== 'X' || board[index] !== '') {
      return
    }

    const nextBoard = board.slice()
    nextBoard[index] = 'X'
    applyBoardState(nextBoard)
    appendLog(`Human placed X at cell ${index}.`)
    setAssistant(null)
    setError(null)
  }

  const handleAssistantMove = async () => {
    if (winner) {
      setError('The game is already over.')
      appendLog('Assistant move skipped because a winner already exists.')
      return
    }
    if (currentPlayer !== 'O') {
      setError('It is still X turn on the GUI board.')
      appendLog('Assistant move rejected because the GUI still expects X.')
      return
    }
    if (selectedProvider === 'ollama' && !ollamaModel.trim()) {
      setError('Choose an Ollama model tag before requesting a move.')
      appendLog('Assistant move rejected because the Ollama model field is empty.')
      return
    }

    setRequestingMove(true)
    setError(null)

    try {
      const response = await fetch('/api/assistant/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          board,
          player: currentPlayer,
          provider: selectedProvider,
          model: selectedProvider === 'ollama' ? ollamaModel.trim() : undefined,
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
      if (!legalMoves(board).includes(data.chosen_move)) {
        throw new Error(`Backend returned illegal move ${data.chosen_move}.`)
      }

      const nextBoard = board.slice()
      nextBoard[data.chosen_move] = 'O'
      applyBoardState(nextBoard)
      setAssistant(data)
      appendLog(`Assistant selected cell ${data.chosen_move} using ${data.provider}:${data.model}.`)
    } catch (requestError) {
      const message = (requestError as Error).message
      setError(message)
      appendLog(`Assistant request failed: ${message}`)
    } finally {
      setRequestingMove(false)
    }
  }

  const handleSendChat = async () => {
    const trimmed = chatDraft.trim()
    if (!trimmed) {
      return
    }
    if (selectedProvider === 'ollama' && !ollamaModel.trim()) {
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
          model: selectedProvider === 'ollama' ? ollamaModel.trim() : undefined,
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
  const statusText = winner
    ? `Winner: ${winner}`
    : boardFull
      ? 'Draw'
      : `Turn: ${currentPlayer}`
  const ollamaOption = providerCatalog?.providers.find((provider) => provider.id === 'ollama')

  return (
    <main className="app-shell">
      <section className="hero-bar">
        <div>
          <p className="eyebrow">Robot Tic-Tac-Toe Lab</p>
          <h1>Observed Input + Visible Reasoning</h1>
        </div>
        <div className="hero-pills">
          <span className="pill">{statusText}</span>
          <span className="pill">Source: GUI board</span>
          <span className="pill">Camera: {vision?.active_source === 'camera' ? 'live' : 'synthetic'}</span>
        </div>
      </section>

      <section className="top-zone">
        <article className="panel board-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Interactive Board</p>
              <h2>GUI-first Game State</h2>
            </div>
            <div className="button-row">
              <button type="button" onClick={resetGame}>
                New Game
              </button>
              <button type="button" onClick={handleAssistantMove} disabled={requestingMove}>
                {requestingMove ? 'Thinking...' : 'Run Assistant Move'}
              </button>
            </div>
          </div>

          <div className="status-strip">
            <div className="status-card">
              <span className="status-label">Operator mode</span>
              <strong>Human X vs Assistant O</strong>
            </div>
            <div className="status-card">
              <span className="status-label">Observed board</span>
              <strong>Direct GUI state</strong>
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
            <img src={`/vision/stream?source=auto&v=${streamNonce}`} alt="Observed input stream" />
            <div className="camera-overlay">
              <span>Preview only</span>
              <span>{vision?.note ?? 'Waiting for backend vision status.'}</span>
            </div>
          </div>

          <div className="status-strip">
            <div className="status-card">
              <span className="status-label">Current source</span>
              <strong>{vision?.active_source ?? 'unknown'}</strong>
            </div>
            <div className="status-card camera-switch-card">
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
            <div className="status-card">
              <span className="status-label">Later role</span>
              <strong>Physical board observation</strong>
            </div>
          </div>
        </article>
      </section>

      <section className="panel assistant-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Assistant</p>
            <h2>Structured Move + Demo Transcript</h2>
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
              <span className="status-label">Ollama Model</span>
              <input
                type="text"
                value={ollamaModel}
                onChange={(event) => setOllamaModel(event.target.value)}
                disabled={selectedProvider !== 'ollama'}
                placeholder="gemma4:e4b"
              />
            </label>
          </div>
        </div>

        <div className="status-strip">
          <div className="status-card">
            <span className="status-label">Selected provider</span>
            <strong>{selectedProvider}</strong>
          </div>
          <div className="status-card">
            <span className="status-label">Model</span>
            <strong>{assistant?.model ?? (selectedProvider === 'ollama' ? ollamaModel : 'mock-strategist-v1')}</strong>
          </div>
          <div className="status-card provider-note">
            <span className="status-label">Provider note</span>
            <strong>
              {selectedProvider === 'ollama'
                ? ollamaOption?.note ?? 'Ollama provider status unavailable.'
                : 'Mock provider is always available for UI testing.'}
            </strong>
          </div>
        </div>

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
            <article className="assistant-card">
              <h3>Observed Board</h3>
              <pre className="mono-panel">{formatBoard(board)}</pre>
            </article>

            <article className="assistant-card">
              <h3>Chosen Move</h3>
              <div className="move-card">
                <span className="move-index">{assistant?.chosen_move ?? '-'}</span>
                <p>{assistant?.explanation ?? 'No assistant move requested yet.'}</p>
                <small>Confidence {assistant ? assistant.confidence.toFixed(2) : '--'}</small>
              </div>
            </article>

            <article className="assistant-card transcript-card">
              <h3>Reasoning Transcript</h3>
              <pre className="mono-panel transcript-panel">
                {assistant?.reasoning_transcript ??
                  'No reasoning transcript yet.\n\nPlace X on the board, then request the assistant move.'}
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
