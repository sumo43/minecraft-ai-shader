import argparse
import cv2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
from mss import mss


class MssScreenCapture:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.running = False
        
        # Create window
        self.root = tk.Tk()
        self.root.title("MSS Screen Capture Test")
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        
        # Add buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.start_button = tk.Button(self.button_frame, text="Start Capture", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.stop_button.config(state=tk.DISABLED)
        
        self.test_button = tk.Button(self.button_frame, text="Capture Once", command=self.capture_once)
        self.test_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.save_button = tk.Button(self.button_frame, text="Capture & Save", command=self.capture_and_save)
        self.save_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Region settings
        if self.args.region:
            self.region = list(map(int, args.region.split(',')))
        else:
            self.region = None
        
        # FPS counter
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_label = tk.Label(self.button_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def capture_screen(self):
        """Capture a region of the screen using mss."""
        try:
            with mss() as sct:
                print(f"Available monitors: {sct.monitors}")
                
                if self.region:
                    x, y, w, h = self.region
                    monitor = {"left": x, "top": y, "width": w, "height": h}
                else:
                    # Capture the first monitor
                    monitor = sct.monitors[1]  # 1 is the first monitor (0 is all monitors combined)
                
                print(f"Capturing from monitor: {monitor}")
                
                # Capture the screen
                sct_img = sct.grab(monitor)
                print(f"Captured image size: {sct_img.size}, mode: {sct_img.pixel_format}")
                
                # Convert to numpy array
                img = np.array(sct_img)
                print(f"Numpy array shape: {img.shape}, dtype: {img.dtype}")
                
                # Resize if needed
                if img.shape[0] != self.height or img.shape[1] != self.width:
                    img = cv2.resize(img, (self.width, self.height))
                    print(f"Resized to: {img.shape}")
                
                # Convert from BGRA to BGR
                if img.shape[2] == 4:  # If it has an alpha channel
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    print("Converted from BGRA to BGR")
                
                return img
        except Exception as e:
            import traceback
            print(f"Error capturing screen: {e}")
            traceback.print_exc()
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def update_display(self, img):
        """Display an image on the canvas."""
        try:
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"Converted to RGB. Shape: {img_rgb.shape}")
            
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            print(f"Created PIL image: {img_pil.size}, mode: {img_pil.mode}")
            
            # Convert to Tkinter PhotoImage
            img_tk = ImageTk.PhotoImage(image=img_pil)
            print("Created Tkinter PhotoImage")
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Display on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            print("Added image to canvas")
            
            # Keep a reference to prevent garbage collection
            self.canvas.image = img_tk
            print("Stored reference to prevent garbage collection")
            
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()
    
    def capture_loop(self):
        """Main capture loop."""
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
            import traceback
            print(f"Error in capture loop: {e}")
            traceback.print_exc()
            
            # Try to continue
            self.root.after(1000, self.capture_loop)
    
    def capture_once(self):
        """Capture a single frame for testing."""
        try:
            # Disable button temporarily
            self.test_button.config(state=tk.DISABLED)
            
            # Capture and display
            img = self.capture_screen()
            self.update_display(img)
            
            # Re-enable button
            self.test_button.config(state=tk.NORMAL)
            
        except Exception as e:
            import traceback
            print(f"Error in capture_once: {e}")
            traceback.print_exc()
            self.test_button.config(state=tk.NORMAL)
    
    def capture_and_save(self):
        """Capture a frame and save it to disk."""
        try:
            # Disable button temporarily
            self.save_button.config(state=tk.DISABLED)
            
            # Capture screen
            img = self.capture_screen()
            
            # Display image
            self.update_display(img)
            
            # Create timestamped filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_capture_{timestamp}.png"
            
            # Save the image
            cv2.imwrite(filename, img)
            print(f"Saved screenshot to {filename}")
            
            # Show success message
            import os
            absolute_path = os.path.abspath(filename)
            self.show_message(f"Saved to:\n{absolute_path}")
            
            # Re-enable button
            self.save_button.config(state=tk.NORMAL)
            
        except Exception as e:
            import traceback
            print(f"Error saving capture: {e}")
            traceback.print_exc()
            self.save_button.config(state=tk.NORMAL)
    
    def show_message(self, message):
        """Show a popup message."""
        message_window = tk.Toplevel(self.root)
        message_window.title("Screenshot Saved")
        
        # Calculate position (center relative to main window)
        window_width = 400
        window_height = 100
        position_x = self.root.winfo_x() + (self.root.winfo_width() - window_width) // 2
        position_y = self.root.winfo_y() + (self.root.winfo_height() - window_height) // 2
        message_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Add message
        label = tk.Label(message_window, text=message, padx=20, pady=20)
        label.pack(expand=True, fill=tk.BOTH)
        
        # Add close button
        close_button = tk.Button(message_window, text="OK", command=message_window.destroy)
        close_button.pack(pady=10)
    
    def start_capture(self):
        """Start continuous capture."""
        if not self.running:
            self.running = True
            self.capture_loop()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.test_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
    
    def stop_capture(self):
        """Stop continuous capture."""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
    
    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="MSS Screen Capture Test")
    
    parser.add_argument('--width', type=int, default=640, 
                      help='Width of the display window')
    parser.add_argument('--height', type=int, default=480, 
                      help='Height of the display window')
    parser.add_argument('--region', type=str, default=None, 
                      help='Region to capture: x,y,width,height')
    parser.add_argument('--fps', type=int, default=30, 
                      help='Target FPS for continuous capture')
    
    args = parser.parse_args()
    
    app = MssScreenCapture(args)
    app.run()


if __name__ == "__main__":
    main()