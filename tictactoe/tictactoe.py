import tkinter as tk
from functools import partial
from dobot_python.dobot import Dobot
from tkinter import PhotoImage
from tkinter import ttk
from helpers.load_calibration import load_calibration
from helpers.robot_motion import RobotMotion
from helpers.game_logic import get_winner_and_cells, is_draw, AIPlayer

# ----------------------------------------------------------- #
# 0.  CONFIGURATION
# ----------------------------------------------------------- #

PORT                        = "/dev/ttyUSB0"
BOARD_ORIENTATION           = 270              # 0/90/180/270 used to rotate board mapping       
APPROACH_OFFSET             = 35             # mm above piece before descending
RETRACT_DISTANCE            = 12             # mm straight up after picking/placing
PLACE_OFFSET                = 8              # mm offset for dropping pieces onto board
POSE_TOL_MM, POSE_POLL_S    = 1.0, 0.05      # mm, s

CAL = load_calibration("calib_points.json", place_offset=PLACE_OFFSET)
PICK_X              = CAL["PICK_X"]
RETURN_X            = CAL["RETURN_X"]
PICK_O              = CAL["PICK_O"]
RETURN_O            = CAL["RETURN_O"]
TTT_CELLS_PICK      = CAL["TTT_CELLS_PICK"]
TTT_CELLS_PLACE     = CAL["TTT_CELLS_PLACE"]

BOARD_SCALE = 3  # integer scaling factor for board images (1 = original size)

# ----------------------------------------------------------- #
# 1. Additional Helpers
# ----------------------------------------------------------- #

def map_gui_to_robot(gui_row: int, gui_col: int, rotation: int = 0):
    """helper to rotate GUI (row,col) to robot (row,col) based on BOARD_ORIENTATION."""
    x, y = gui_col, 2 - gui_row

    if   rotation == 0:   robot_r, robot_c = y,         x
    elif rotation == 90:  robot_r, robot_c = 2 - x,     y
    elif rotation == 180: robot_r, robot_c = 2 - y, 2 - x
    elif rotation == 270: robot_r, robot_c = x,      2 - y
    else:
        raise ValueError("rotation must be 0, 90, 180, or 270")

    return robot_r, robot_c


# ----------------------------------------------------------- #
# 2. TKINTER GUI (Robot Tic-Tac-Toe)
# ----------------------------------------------------------- #

class TicTacToeGUI(tk.Tk):
    def __init__(self, port = '/dev/ttyUSB0', vel=50.0, acc=50.0):
        super().__init__()
        self.robot = Dobot(port, vel, acc)
        self.robot_motions = RobotMotion(self.robot, approach_offset=APPROACH_OFFSET, retract_distance=RETRACT_DISTANCE, pose_tol_mm=POSE_TOL_MM, pose_poll_s=POSE_POLL_S)
        self.title("Robot Tic-Tac-Toe")
        self.geometry("960x1280")
        self.resizable(False, False)
        
        # Colors for updating labels
        self.PLAYER_COLORS = {'X': 'red', 'O': 'blue'}
        
        # Load images for the buttons (empty cells, X, O, rotation arrows)
        self.load_images()

        # Initialize game variables
        self.board = [["", "", ""],
                      ["", "", ""],
                      ["", "", ""]]
        
        self.current_player = "X" # Game always starts with player X
        self.game_over = False
        self.busy = False

        # Initialize result counters
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0

        # Keeping count of what round it is for movement routines
        self.x_picked = 0
        self.o_picked = 0

        # Initialize AI settings
        self.game_mode = tk.StringVar(value="PvP")
        self.ai_difficulty = tk.StringVar(value="easy")
        self.ai2_difficulty = tk.StringVar(value="easy")
        self.ai_player = AIPlayer(player='O', difficulty=self.ai_difficulty.get())
        self.ai2_player = AIPlayer(player='X', difficulty=self.ai2_difficulty.get())
        self.aivai_active = False
        self._after_ids = set()

        # Build UI
        self.create_top_area()
        self.create_settings_ui()
        self.create_message_area()
        self.create_main_board()
        self.create_bottom_area()

    # ----------------------
    #    UI CREATION
    # ----------------------
    def load_images(self):
        try:
            base_empty = PhotoImage(file="assets/empty.png")
            base_x     = PhotoImage(file="assets/x.png")
            base_o     = PhotoImage(file="assets/o.png")

            # Scale board tiles by BOARD_SCALE (must be an int)
            s = BOARD_SCALE
            if s > 1:
                self.empty_cell_img = base_empty.zoom(s, s)
                self.x_img          = base_x.zoom(s, s)
                self.o_img          = base_o.zoom(s, s)
            else:
                # fallback to original size if scale = 1
                self.empty_cell_img = base_empty
                self.x_img          = base_x
                self.o_img          = base_o

        except tk.TclError as e:
            print(f"Error loading images: {e}")
            self.empty_cell_img = None
            self.x_img = None
            self.o_img = None


    def create_top_area(self):
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(pady=5)

        # Current Player Display
        self.player_frame = tk.Frame(self.top_frame)
        self.player_frame.pack(pady=5)

        self.current_player_text = tk.Label(self.player_frame, text="Current Player: ", font=("Impact", 28))
        self.current_player_text.pack(side=tk.LEFT)

        self.current_player_symbol = tk.Label(self.player_frame, text="X", font=("Helvetica", 28, "bold"), fg='red')
        self.current_player_symbol.pack(side=tk.LEFT)
        self.update_current_player_label()

        # Separator
        separator = ttk.Separator(self.top_frame, orient='horizontal')
        separator.pack(fill='x', pady=5)

        # Results Frame
        self.results_frame = tk.Frame(self.top_frame)
        self.results_frame.pack(pady=5)

        self.results_label = tk.Label(self.results_frame, text=self.get_results_text(), font=("Helvetica", 18, "bold"))
        self.results_label.pack(side=tk.LEFT, padx=5)

        self.reset_stats_button = tk.Button(self.results_frame, text="Reset Stats", command=self.reset_stats)
        self.reset_stats_button.pack(side=tk.LEFT, padx=10)

        separator2 = ttk.Separator(self.top_frame, orient='horizontal')
        separator2.pack(fill='x', pady=5)

    def create_settings_ui(self):
        self.settings_frame = tk.Frame(self)
        self.settings_frame.pack(pady=5)

        # Game Mode / Difficulty
        settings_side_frame = tk.Frame(self.settings_frame)
        settings_side_frame.pack()

        # Game mode frame
        game_mode_frame = tk.Frame(settings_side_frame)
        game_mode_frame.grid(row=0, column=0, padx=10, pady=2, sticky='nw')

        game_mode_label = tk.Label(game_mode_frame, text="Select Game Mode:", font=("Helvetica", 11, "bold"))
        game_mode_label.pack(anchor='w')

        modes = [("Player vs Player", "PvP"), ("Player vs AI", "PvAI"), ("AI vs AI", "AivAI")]
        for text, mode in modes:
            rb = tk.Radiobutton(game_mode_frame, text=text, variable=self.game_mode, value=mode, command=self.on_mode_change)
            rb.pack(anchor='w')

        # AI1 difficulty frame
        self.ai_difficulty_frame = tk.Frame(settings_side_frame)
        self.ai_difficulty_frame.grid(row=0, column=1, padx=10, pady=2, sticky='nw')

        ai_difficulty_label = tk.Label(self.ai_difficulty_frame, text="AI Difficulty (O)", font=("Helvetica", 11, "bold"))
        ai_difficulty_label.pack(anchor='w')

        difficulties = [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]
        for text, difficulty in difficulties:
            rb = tk.Radiobutton(self.ai_difficulty_frame, text=text, variable=self.ai_difficulty,
                                 value=difficulty, command=self.on_o_difficulty_change)
            rb.pack(anchor='w')

        # AI2 difficulty frame (hidden unless AivAI)
        self.ai_difficulty_secondary_frame = tk.Frame(settings_side_frame)
        ai_difficulty_secondary_label = tk.Label(self.ai_difficulty_secondary_frame, text="AI Difficulty (X)", font=("Helvetica", 11, "bold"))
        ai_difficulty_secondary_label.pack(anchor='w')

        for text, difficulty in difficulties:
            rb = tk.Radiobutton(self.ai_difficulty_secondary_frame, text=text, variable=self.ai2_difficulty,
                                 value=difficulty, command=self.on_x_difficulty_change)
            rb.pack(anchor='w')

        self.ai_difficulty_secondary_frame.grid(row=0, column=2, padx=10, pady=2, sticky='nw')
        self.ai_difficulty_secondary_frame.grid_remove()

        separator3 = ttk.Separator(self.settings_frame, orient='horizontal')
        separator3.pack(fill='x', pady=3)

        # Toggle AI settings based on initial mode
        self.toggle_ai_settings()

    def create_message_area(self):
        self.message_frame = tk.Frame(self)
        self.message_frame.pack(pady=5)
        self.result_label = tk.Label(self.message_frame, text="", font=("Helvetica", 16))
        self.result_label.pack()

    def create_main_board(self):
        self.main_board_frame = tk.Frame(self)
        self.main_board_frame.pack(pady=3)

        # Board frame
        self.board_frame = tk.Frame(self.main_board_frame)
        self.board_frame.pack(side=tk.LEFT, padx=5)

        self.buttons = []
        for r in range(3):
            row_buttons = []
            for c in range(3):
                btn = tk.Button(self.board_frame, image=self.empty_cell_img, command=partial(self.cell_clicked, r, c))
                btn.grid(row=r, column=c, padx=2, pady=2)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def create_bottom_area(self):
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(pady=0)

        control_frame = tk.Frame(self.bottom_frame)
        control_frame.pack(pady=8)

        reset_button = tk.Button(control_frame, text="New Game", command=self.reset_game, width=10)
        reset_button.pack(side=tk.LEFT, padx=5)

        cleanup_button = tk.Button(control_frame, text="Cleanup Board", command=self.cleanup_on_button, width=12)
        cleanup_button.pack(side=tk.LEFT, padx=5)

        start_button = tk.Button(control_frame, text="Start Game", command=self.start_game, width=10)
        start_button.pack(side=tk.LEFT, padx=5)

    # ----------------------
    #   ROBOT / GAME LOGIC
    # ----------------------

    # -- scheduling helpers for AIvAI timers ---------------------------------
    def _after(self, ms, func):
        """Schedule and remember the 'after' ID for later cancellation."""
        aid = self.after(ms, func)
        self._after_ids.add(aid)
        return aid

    def _cancel_afters(self):
        """Cancel and clear all remembered 'after' timers (safe to call anytime)."""
        for aid in list(self._after_ids):
            try:
                self.after_cancel(aid)
            except Exception:
                pass
            self._after_ids.discard(aid)

    def attempt_move(self, row, col, piece):
        """
        Consolidates the logic for making a move:
          1) Checks if cell is empty
          2) Robot picks & places the piece
          3) Updates board and button images
          4) Checks for winner/draw
          5) Switches player or ends game
        Returns True if move succeeded, False if invalid.
        """
        if self.board[row][col] != "":
            self.update_status("Invalid move. Try again.", transient=True)
            return False

        # Update board first
        self.board[row][col] = piece

        # Robot picks up appropriate piece
        if piece == "X":
            self.x_picked += 1
            if self.x_picked == 4:
                self.robot_motions.special_pick(PICK_X)
            else:
                self.robot_motions.pick_object(PICK_X, mode='pickup')
        else:   # 'O'
            self.o_picked += 1
            if self.o_picked == 4:
                self.robot_motions.special_pick(PICK_O)
            else:
                self.robot_motions.pick_object(PICK_O, mode='pickup')

        # Map the (row, col) to robot's orientation
        robot_r, robot_c = map_gui_to_robot(row, col, BOARD_ORIENTATION)
        self.robot_motions.place_object(TTT_CELLS_PLACE[robot_r][robot_c])
        # Update button image in GUI
        if piece == "X":
            self.buttons[row][col].config(image=self.x_img)
        else:
            self.buttons[row][col].config(image=self.o_img)

        # Check winner or draw
        winner, winning_cells = get_winner_and_cells(self.board)
        if winner:
            self.game_over = True
            self.highlight_winning_line(winning_cells)
            self.show_result(f"Player {winner} wins!")
        elif is_draw(self.board):
            self.game_over = True
            self.show_result("It's a Draw!")
        else:
            # Switch current player
            self.current_player = "O" if self.current_player == "X" else "X"
            self.update_current_player_label()

        return True

    def cleanup_board(self):
        """
        Robot collects each piece from the board (considering rotation) and
        returns them to the respective slide. Resets board state.
        """
        for r in range(3):
            for c in range(3):
                piece = self.board[r][c]
                if piece != "":
                    # Map (r, c) to robot coords
                    robot_r, robot_c = map_gui_to_robot(r, c, BOARD_ORIENTATION)
                    pick_position = TTT_CELLS_PICK[robot_r][robot_c]

                    if piece == "X":
                        self.robot_motions.pick_object(pick_position, mode='cleanup')
                        self.robot_motions.place_object(RETURN_X)
                    else:  # 'O'
                        self.robot_motions.pick_object(pick_position, mode='cleanup')
                        self.robot_motions.place_object(RETURN_O)
        # Reset board
        self.board = [["", "", ""],
                      ["", "", ""],
                      ["", "", ""]]

    # ----------------------
    #      EVENT HANDLERS
    # ----------------------

    def cell_clicked(self, row, col):
        """Human clicks a square (PvP or PvAI)."""
        if self.game_over or self.busy or self.game_mode.get() == "AivAI":
            return
        if self.board[row][col] != "":
            self.update_status("Invalid move. Try again.", transient=True)
            return

        # Mark GUI as busy & disable clickables
        self.busy = True
        self.disable_board_buttons()
        self.after(10, lambda r=row, c=col: self._human_move(r, c))

    def _human_move(self, row, col):
        """Runs one human move synchronously, then decides who goes next."""
        self.attempt_move(row, col, self.current_player)

        if self.game_over:
            self.busy = False
            return

        if self.game_mode.get() == "PvP":
            # Give Tk one idle cycle before re‑enabling to avoid race clicks
            self.after_idle(self._release_human_turn)

        elif self.game_mode.get() == "PvAI" and self.current_player == "O":
            # Still busy: AI + robot will play now
            self.after(80, self.ai_move)        # small delay lets GUI refresh
        else:
            # Shouldn’t happen, but reset just in case
            self.busy = False
            self.enable_board_buttons()

    def _release_human_turn(self):
        """Enables board after robot has certainly finished the previous move."""
        self.busy = False
        if not self.game_over:
            self.enable_board_buttons()

    def ai_move(self):
        if self.game_over:
            return
        move = self.ai_player.get_move(self.board)
        if move is None:
            self.update_status("AI has no moves. Board full or error.")
            self.busy = False
            self.enable_board_buttons()
            return

        r, c = move
        self.attempt_move(r, c, self.ai_player.player)

        # AI finished; give control back to human X
        self.busy = False
        if not self.game_over:
            self.enable_board_buttons()

    def start_game(self):
        """
        Called when pressing 'Start Game'. Initiates AivAI or resets for PvP/PvAI.
        """
        mode = self.game_mode.get()

        # Always stop any running AIvAI timers before switching/starting
        self._cancel_afters()

        if mode in ["PvP", "PvAI"]:
            # Leaving AIvAI -> mark inactive and return to human-driven flow
            if self.aivai_active:
                self.aivai_active = False
            self.reset_game()
            self.enable_board_buttons()
            self.update_status(f"{mode} mode selected.")
        elif mode == "AivAI":
            if not self.aivai_active:
                self.reset_game()
                self.disable_board_buttons()
                self.update_status("AI vs AI game started.", transient=True)
                self._after(1000, self.start_aivai_game)  # <-- use wrapper
        else:
            self.show_result("Please select a valid game mode.")


    def start_aivai_game(self):
        """
        Initiates the AI vs AI game loop.
        """
        if self.aivai_active:
            return
        self.aivai_active = True
        self.disable_board_buttons()
        self.update_status("AI vs AI game started.", transient=True)
        self._after(1000, self.aivai_move)  # <-- use wrapper


    def aivai_move(self):
        """
        AI vs AI move sequence (ticks via Tk 'after').
        """
        # Stop if mode changed or manually deactivated
        if not self.aivai_active or self.game_mode.get() != "AivAI":
            return

        if self.game_over:
            self._after(2000, self.cleanup_board_automatically)
            self.aivai_active = False
            return

        current_ai = self.ai2_player if self.current_player == "X" else self.ai_player
        move = current_ai.get_move(self.board)
        if move:
            r, c = move
            self.attempt_move(r, c, current_ai.player)
            self.update()  # Force GUI refresh for smoothness

            if not self.game_over and self.aivai_active and self.game_mode.get() == "AivAI":
                self._after(10, self.aivai_move)  # schedule next tick
        else:
            self.update_status(f"AI {current_ai.player} has no moves.", transient=True)


    # ----------------------
    #      UI UPDATES
    # ----------------------

    def highlight_winning_line(self, winning_cells):
        """
        Highlights the three winning cells in green.
        """
        if not winning_cells:
            return
        for (r, c) in winning_cells:
            self.buttons[r][c].config(bg="lightgreen")

    def update_current_player_label(self):
        color = self.PLAYER_COLORS.get(self.current_player, 'black')
        self.current_player_symbol.config(text=self.current_player, fg=color)

    def update_status(self, message, transient=False, delay=2000):
        """
        Updates the status message. If transient, clears after a delay.
        """
        self.result_label.config(text=message)
        if transient:
            self.after(delay, lambda: self.result_label.config(text=""))

    def show_result(self, message):
        """
        Displays final result and updates stats if needed.
        """
        self.result_label.config(text=message)
        if "wins" in message:
            # e.g., "Player X wins!"
            winner = message.split(" ")[1]
            self.update_results(winner)
        elif "Draw" in message:
            self.update_results("Draw")

        # In AI vs AI, auto-clean & restart
        if self.game_mode.get() == "AivAI":
            self._after(2000, self.cleanup_board_automatically)


    def toggle_ai_settings(self):
        """
        Enables or disables AI difficulty frames based on game mode.
        """
        mode = self.game_mode.get()
        if mode == "PvAI":
            # Enable AI1, hide AI2
            for child in self.ai_difficulty_frame.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    child.config(state=tk.NORMAL)
            self.ai_difficulty_secondary_frame.grid_remove()
        elif mode == "AivAI":
            # Enable both AI difficulties
            for child in self.ai_difficulty_frame.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    child.config(state=tk.NORMAL)
            for child in self.ai_difficulty_secondary_frame.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    child.config(state=tk.NORMAL)
            self.ai_difficulty_secondary_frame.grid()
        else:
            # Disable & hide AI difficulties
            self.ai_difficulty.set("easy")
            self.ai2_difficulty.set("easy")
            for child in self.ai_difficulty_frame.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    child.config(state=tk.DISABLED)
            for child in self.ai_difficulty_secondary_frame.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    child.config(state=tk.DISABLED)
            self.ai_difficulty_secondary_frame.grid_remove()

    def on_mode_change(self):
        """
        Whenever the game mode (PvP / PvAI / AivAI) changes,
        we treat it as starting a brand new game in that mode.
        """
        # Stop any scheduled AI timers (AIvAI loop etc.)
        self._cancel_afters()
        self.aivai_active = False
        self.busy = False

        mode = self.game_mode.get()

        # Start a fresh game state for the new mode
        self.reset_game()

        # For AI vs AI, humans shouldn't click the board at all
        if mode == "AivAI":
            self.disable_board_buttons()
        else:
            self.enable_board_buttons()

        # Update which AI difficulty controls are visible/enabled
        self.toggle_ai_settings()

        # Small status info
        self.update_status(f"{mode} mode selected. New game started.", transient=True)



    def on_o_difficulty_change(self):
        difficulty = self.ai_difficulty.get()
        self.ai_player = AIPlayer(player='O', difficulty=difficulty)
        self.update_status(f"AI O difficulty set to {difficulty}", transient=True)

    def on_x_difficulty_change(self):
        # Only relevant if the game mode is AivAI, but you can decide how to handle it
        difficulty2 = self.ai2_difficulty.get()
        self.ai2_player = AIPlayer(player='X', difficulty=difficulty2)
        self.update_status(f"AI X difficulty set to {difficulty2}", transient=True)
   

    # ----------------------
    #    GAME MANAGEMENT
    # ----------------------

    def reset_game(self):
        """
        Resets the game state (board, UI, AI memo).
        """
        self._cancel_afters()
        self.aivai_active = False
        self.enable_board_buttons()
        self.game_over = False
        self.board = [["", "", ""], ["", "", ""], ["", "", ""]]
        self.current_player = "X"

        # Clear AI memo
        self.ai_player.memo = {}
        self.ai2_player.memo = {}

        # Reset pick counts
        self.x_picked = 0
        self.o_picked = 0

        # Clear result message & update labels
        self.result_label.config(text="")
        self.update_current_player_label()

        # Reset button images and backgrounds
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(image=self.empty_cell_img, bg=self.cget("bg"))
        self.update_status("New game started.", transient=True)

    def cleanup_on_button(self):
        """
        Called when the user presses 'Cleanup Board' button.
        """
        self.cleanup_board()
        self.reset_game()

    def cleanup_board_automatically(self):
        """
        Cleans up the board after AI vs AI finishes, then restarts AI vs AI (if still in that mode).
        """
        self.cleanup_board()
        self.reset_game()
        self.aivai_active = False
        if self.game_mode.get() == "AivAI":
            self._after(2000, self.start_aivai_game)


    def get_results_text(self):
        return f"X Wins: {self.x_wins} | O Wins: {self.o_wins} | Draws: {self.draws}"

    def update_results(self, winner):
        if winner == "X":
            self.x_wins += 1
        elif winner == "O":
            self.o_wins += 1
        elif winner == "Draw":
            self.draws += 1
        self.results_label.config(text=self.get_results_text())

    def reset_stats(self):
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        self.results_label.config(text=self.get_results_text())
        self.update_status("Player stats cleared.", transient=True)

    def disable_board_buttons(self):
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)

    def enable_board_buttons(self):
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.NORMAL)

# ----------------------------------------------------------- #
# 6. MAIN LAUNCH
# ----------------------------------------------------------- #

def gui_main():
    app = TicTacToeGUI(port=PORT, vel=200, acc=150)
    app.mainloop()

if __name__ == "__main__":
    gui_main()
