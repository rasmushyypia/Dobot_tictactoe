import cv2
import numpy as np
import os
from pathlib import Path

def detect_and_crop_grid(image_path, output_dir):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing to find the hashtag lines
    # 1. Blur to remove noise (notebook dots)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Adaptive threshold to handle lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Use morphology to strengthen lines
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # 4. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # For a hand-drawn grid, the "hashtags" are often the largest contour
    # or we can look for the bounding box of all marks.
    # In a lab setting, we might just assume the grid fills most of the frame
    # or use HoughLinesP to find the 4 main lines.
    
    # Simplified approach for POC: 
    # Just divide the image into a 3x3 grid based on the full frame
    # (assuming the user has centered the board)
    h, w = img.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    
    os.makedirs(output_dir, exist_ok=True)
    
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            cell_img = img[y1:y2, x1:x2]
            cell_path = Path(output_dir) / f"cell_{idx}.jpg"
            cv2.imwrite(str(cell_path), cell_img)
            print(f"Saved {cell_path}")

if __name__ == "__main__":
    sample_img = r"C:\Users\rasmu\Documents\0_CODING\ristinolla_ai\debug_images\web_ttt_20260405_205608_767_camera_frame_analysis.jpg"
    detect_and_crop_grid(sample_img, "debug_images/poc_crops")
