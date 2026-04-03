const boardGrid = document.getElementById("board-grid");
const newGameBtn = document.getElementById("new-game-btn");
const assistantMoveBtn = document.getElementById("assistant-move-btn");
const reasoningText = document.getElementById("reasoning-text");
const observedBoardText = document.getElementById("observed-board-text");
const moveIndex = document.getElementById("move-index");
const moveSummary = document.getElementById("move-summary");
const eventLog = document.getElementById("event-log");
const turnPill = document.getElementById("turn-pill");
const sourcePill = document.getElementById("source-pill");
const startCameraBtn = document.getElementById("start-camera-btn");
const stopCameraBtn = document.getElementById("stop-camera-btn");
const cameraVideo = document.getElementById("camera-video");
const cameraFallback = document.getElementById("camera-fallback");
const cameraStateTag = document.getElementById("camera-state-tag");

let board = Array(9).fill("");
let currentPlayer = "X";
let activeCameraStream = null;

function winningLines() {
  return [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
}

function logEvent(message) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
  eventLog.prepend(item);
  while (eventLog.children.length > 7) {
    eventLog.removeChild(eventLog.lastChild);
  }
}

function getWinner(cells) {
  for (const line of winningLines()) {
    const [a, b, c] = line;
    if (cells[a] && cells[a] === cells[b] && cells[b] === cells[c]) {
      return cells[a];
    }
  }
  return null;
}

function legalMoves(cells) {
  return cells
    .map((value, index) => (value === "" ? index : -1))
    .filter((index) => index >= 0);
}

function formatBoard(cells) {
  const rows = [];
  for (let row = 0; row < 3; row += 1) {
    const values = cells
      .slice(row * 3, row * 3 + 3)
      .map((cell) => (cell === "" ? "." : cell));
    rows.push(values.join(" "));
  }
  return rows.join("\n");
}

function renderBoard() {
  boardGrid.innerHTML = "";
  board.forEach((value, index) => {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = `cell ${value === "X" ? "cell-x" : ""} ${value === "O" ? "cell-o" : ""}`.trim();
    cell.textContent = value;
    cell.setAttribute("aria-label", `Cell ${index}`);
    cell.addEventListener("click", () => handleHumanMove(index));
    boardGrid.appendChild(cell);
  });

  turnPill.textContent = `Turn: ${currentPlayer}`;
  observedBoardText.textContent = formatBoard(board);
}

function resetAssistantState() {
  reasoningText.textContent = "Assistant idle.\n\nUse the board above to set up a position, then run the assistant move.";
  moveIndex.textContent = "-";
  moveSummary.textContent = "No move requested yet.";
}

function newGame() {
  board = Array(9).fill("");
  currentPlayer = "X";
  renderBoard();
  resetAssistantState();
  logEvent("Started a new GUI board session.");
}

function handleHumanMove(index) {
  if (currentPlayer !== "X" || board[index] !== "" || getWinner(board)) {
    return;
  }

  board[index] = "X";
  currentPlayer = getWinner(board) || legalMoves(board).length === 0 ? "X" : "O";
  renderBoard();
  logEvent(`Human placed X at cell ${index}.`);

  if (getWinner(board)) {
    reasoningText.textContent = "Game over: X already has a winning line.";
    moveSummary.textContent = "Assistant turn skipped because the game is over.";
  }
}

function chooseHeuristicMove(cells, player) {
  const opponent = player === "X" ? "O" : "X";
  const open = legalMoves(cells);

  for (const move of open) {
    const next = [...cells];
    next[move] = player;
    if (getWinner(next) === player) {
      return { move, reason: `I can win immediately by placing ${player} at ${move}.` };
    }
  }

  for (const move of open) {
    const next = [...cells];
    next[move] = opponent;
    if (getWinner(next) === opponent) {
      return { move, reason: `I need to block ${opponent} from winning, so I choose ${move}.` };
    }
  }

  if (open.includes(4)) {
    return { move: 4, reason: "The center is open, so I take it to control the board." };
  }

  const corners = open.filter((move) => [0, 2, 6, 8].includes(move));
  if (corners.length > 0) {
    return { move: corners[0], reason: `The center is unavailable, so I take corner ${corners[0]}.` };
  }

  return { move: open[0], reason: `No tactical threat detected. I take the next legal cell ${open[0]}.` };
}

function runAssistantMove() {
  if (getWinner(board)) {
    reasoningText.textContent = "Observed board already contains a winner.\n\nNo assistant move will be executed.";
    moveIndex.textContent = "-";
    moveSummary.textContent = "Game over.";
    logEvent("Assistant move rejected because the game is already over.");
    return;
  }

  if (currentPlayer !== "O") {
    reasoningText.textContent = "It is not the assistant turn yet.\n\nWaiting for a human X move on the GUI board.";
    moveIndex.textContent = "-";
    moveSummary.textContent = "Waiting for X.";
    logEvent("Assistant is waiting because the GUI board still expects X.");
    return;
  }

  const open = legalMoves(board);
  if (open.length === 0) {
    reasoningText.textContent = "The board is full.\n\nNo legal moves remain.";
    moveIndex.textContent = "-";
    moveSummary.textContent = "Draw state.";
    logEvent("Assistant found no legal moves.");
    return;
  }

  const decision = chooseHeuristicMove(board, "O");
  const transcript = [
    "Mock reasoning transcript for the current prototype.",
    "",
    "Observed source: interactive GUI board",
    `Observed board:\n${formatBoard(board)}`,
    "",
    `Legal moves: ${open.join(", ")}`,
    decision.reason,
    "",
    `Final structured move: ${decision.move}`,
  ].join("\n");

  reasoningText.textContent = transcript;
  moveIndex.textContent = String(decision.move);
  moveSummary.textContent = "Prototype output: this move is separate from the visible reasoning transcript.";

  board[decision.move] = "O";
  currentPlayer = getWinner(board) || legalMoves(board).length === 0 ? "O" : "X";
  renderBoard();
  logEvent(`Assistant selected cell ${decision.move}.`);
}

async function startCamera() {
  if (activeCameraStream) {
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    activeCameraStream = stream;
    cameraVideo.srcObject = stream;
    cameraVideo.style.display = "block";
    cameraFallback.style.display = "none";
    cameraStateTag.textContent = "Camera Live";
    sourcePill.textContent = "Source: GUI Board + Camera Preview";
    logEvent("Started browser camera preview.");
  } catch (error) {
    cameraStateTag.textContent = "Camera Blocked";
    logEvent(`Camera preview failed: ${error.message}`);
  }
}

function stopCamera() {
  if (activeCameraStream) {
    activeCameraStream.getTracks().forEach((track) => track.stop());
  }
  activeCameraStream = null;
  cameraVideo.srcObject = null;
  cameraVideo.style.display = "none";
  cameraFallback.style.display = "grid";
  cameraStateTag.textContent = "Camera Off";
  sourcePill.textContent = "Source: GUI Board";
  logEvent("Stopped browser camera preview.");
}

newGameBtn.addEventListener("click", newGame);
assistantMoveBtn.addEventListener("click", runAssistantMove);
startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);

window.addEventListener("beforeunload", stopCamera);

newGame();

