import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Test script to confirm display capabilities

def main():
    # Create window
    root = tk.Tk()
    root.title("Display Test")
    root.geometry("1280x720")
    
    # Create canvas
    canvas = tk.Canvas(root, width=1280, height=720)
    canvas.pack()
    
    # Create a test image (colorful gradient)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Create a gradient
    for i in range(720):
        for j in range(1280):
            img[i, j, 0] = i * 255 // 720  # Blue gradient
            img[i, j, 1] = j * 255 // 1280  # Green gradient
            img[i, j, 2] = ((i + j) * 255) // (720 + 1280)  # Red gradient
    
    # Convert to PIL format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Display the image
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Keep a reference
    
    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()