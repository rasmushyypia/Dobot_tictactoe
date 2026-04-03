import pyautogui
import requests
import time
import base64
import re
import json

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b"

# Coordinates (Verify these)
GRID_X, GRID_Y, GRID_W, GRID_H = (540, 337, 350, 350) 
VISION_PADDING_TOP = 120 
VISION_REGION = (GRID_X, GRID_Y - VISION_PADDING_TOP, GRID_W, GRID_H + VISION_PADDING_TOP)

def get_move_from_gemma(image_path):
    # Much stricter persona: Only allowed to move if it's O's turn.
    prompt = (
        "TASK: You are the 'O' player in Tic-Tac-Toe.\n"
        "1. LOOK: What does the text at the top say? 'Player 1 (X)' or 'Player 2 (O)'?\n"
        "2. RULE: You are ONLY allowed to move if it is Player 2 (O) turn.\n"
        "3. OUTPUT FORMAT:\n"
        "   - If it is Player 1 (X) turn: Output 'STATUS: WAITING_FOR_X'\n"
        "   - If it is Player 2 (O) turn: Output 'STATUS: O_TURN_MOVE [index]'\n"
        "   - If the game is OVER: Output 'STATUS: GAME_OVER'\n"
        "Think step-by-step, but you MUST end with one of those three exact STATUS lines."
    )
    
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "images": [img_data], "stream": True 
        }, stream=True)
        
        full_response = ""
        print("\n" + "="*40 + "\n👀 MONITORING BOARD...")
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("response", "")
                full_response += content
                print(content, end="", flush=True)
        print("\n" + "="*40)
        return full_response
    except Exception as e:
        print(f"Error: {e}")
        return ""

def click_grid_index(index):
    col = index % 3
    row = index // 3
    target_x = GRID_X + (col * (GRID_W/3)) + (GRID_W/6)
    target_y = GRID_Y + (row * (GRID_H/3)) + (GRID_H/6)
    
    # Quick, precise move
    pyautogui.moveTo(target_x, target_y, duration=0.2)
    pyautogui.click()
    print(f"🎯 O-PLAYER MOVED TO INDEX {index}")

def main():
    print("🚀 Agent Active. O-Bot is waiting for X-Human.")
    
    while True:
        screenshot = pyautogui.screenshot(region=VISION_REGION)
        screenshot.save("agent_view.png")
        
        analysis = get_move_from_gemma("agent_view.png")
        
        # 1. Game Over Guard
        if "GAME_OVER" in analysis.upper():
            print("🏁 Game has ended. Standing by for reset...")
            time.sleep(5)
            continue

        # 2. The Turn Gatekeeper (The most important fix)
        # We look specifically for 'O_TURN_MOVE' to ensure it's not moving for X
        if "O_TURN_MOVE" in analysis.upper():
            # Extract the number that comes immediately after O_TURN_MOVE
            match = re.search(r"O_TURN_MOVE\s*[:=\[ ]*(\d)", analysis, re.IGNORECASE)
            if match:
                move_idx = int(match.group(1))
                click_grid_index(move_idx)
                print("✅ O's move complete. Your turn, X!")
                time.sleep(5) 
            else:
                print("⚠️ O's turn, but index was garbled. Retrying...")
        else:
            # If it's WAITING_FOR_X or just talking, we DO NOT click.
            print("🕒 Standing by... It is Player 1 (X)'s turn.")
            time.sleep(3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Deactivated.")