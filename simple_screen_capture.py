import argparse
import cv2
import numpy as np
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
from mss import mss

class SimpleScreenCapture:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.running = False
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Simple Screen Capture")
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        
        # Add buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # FPS counter
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_label = tk.Label(self.button_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def capture_screen(self):
        try:
            # Use mss to capture the screen
            print("Capturing screen with mss")
            with mss() as sct:
                if self.args.region:
                    x, y, w, h = map(int, self.args.region.split(','))
                    monitor = {"left": x, "top": y, "width": w, "height": h}
                else:
                    # Capture the first monitor
                    monitor = sct.monitors[1]  # 1 is the first monitor (0 is all monitors)
                
                # Capture the screen
                sct_img = sct.grab(monitor)
                
                # Convert to numpy array
                img = np.array(sct_img)
                
                # Resize if needed
                if img.shape[0] != self.height or img.shape[1] != self.width:
                    img = cv2.resize(img, (self.width, self.height))
                
                # Convert from BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                print(f"Captured screen with shape: {img.shape}, dtype: {img.dtype}")
                return img
        except Exception as e:
            import traceback
            print(f"Error capturing screen: {e}")
            traceback.print_exc()
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def update_display(self, img):
        try:
            print(f"Updating display with image shape: {img.shape}, dtype: {img.dtype}")
            print(f"Image min: {img.min()}, max: {img.max()}")
            
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            print("Created PIL image")
            
            # Convert to Tkinter PhotoImage
            img_tk = ImageTk.PhotoImage(image=img_pil)
            print("Created Tk image")
            
            # Display on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            print("Added to canvas")
            
            # Keep a reference to prevent garbage collection
            self.canvas.image = img_tk
            print("Stored reference")
            
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()
    
    def capture_loop(self):
        if not self.running:
            return
        
        try:
            start_time = time.time()
            
            # Capture screen
            img = self.capture_screen()
            
            # Update display
            self.update_display(img)
            
            # Update FPS counter
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Calculate delay
            elapsed = time.time() - start_time
            delay = max(1, int((1.0/self.args.fps - elapsed) * 1000))
            
            # Schedule next frame
            self.root.after(delay, self.capture_loop)
        
        except Exception as e:
            print(f"Error in capture loop: {e}")
            # Try to continue
            self.root.after(1000, self.capture_loop)
    
    def start_capture(self):
        if not self.running:
            self.running = True
            self.capture_loop()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
    
    def stop_capture(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def on_close(self):
        self.running = False
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Simple Screen Capture")
    
    parser.add_argument('--width', type=int, default=1280, 
                       help='Width of the display window')
    parser.add_argument('--height', type=int, default=720, 
                       help='Height of the display window')
    parser.add_argument('--region', type=str, default=None, 
                       help='Region to capture: x,y,width,height')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target FPS')
    
    args = parser.parse_args()
    
    app = SimpleScreenCapture(args)
    app.run()

if __name__ == "__main__":
    main()