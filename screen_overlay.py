import argparse
import cv2
import numpy as np
import os
import torch
import time
import pyautogui
import threading
import win32gui
import win32con
import win32api
from PIL import Image, ImageTk
import tkinter as tk
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_screen import RealESRGANer


def setup_upsampler(args):
    """Set up the RealESRGAN upsampler."""
    # Select the appropriate model architecture based on the model name
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    else:
        # Default to a simple model suitable for real-time processing
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, act_type='prelu')
        netscale = 1

    # Determine model path
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f"Model path {model_path} does not exist. Please download the model first.")

    # Initialize the upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    
    # Compile the model for better performance
    upsampler.model = torch.compile(upsampler.model, fullgraph=True, dynamic=False, mode="reduce-overhead")
    
    return upsampler


class TransparentOverlay:
    def __init__(self, x, y, width, height):
        # Create a window with transparent background
        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
            "STATIC", "Overlay", win32con.WS_POPUP,
            x, y, width, height, 0, 0, 0, None
        )
        # Set the window transparency
        win32gui.SetLayeredWindowAttributes(
            self.hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY
        )
        # Show the window
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        
        # Create device context for drawing
        self.hdc = win32gui.GetDC(self.hwnd)
        self.mem_dc = win32gui.CreateCompatibleDC(self.hdc)
        self.bitmap = win32gui.CreateCompatibleBitmap(self.hdc, width, height)
        win32gui.SelectObject(self.mem_dc, self.bitmap)
        
        self.width = width
        self.height = height
        
    def update(self, image):
        # Convert OpenCV image to a format that can be used with Windows GDI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # Create a Windows bitmap from PIL image
        hbmp = pil_image.tobitmap()
        
        # Draw the image to our memory DC
        win32gui.BitBlt(self.mem_dc, 0, 0, self.width, self.height, 
                        win32gui.GetDC(win32gui.GetDesktopWindow()), 0, 0, win32con.SRCCOPY)
        
        # Blend the image
        win32gui.AlphaBlend(self.mem_dc, 0, 0, self.width, self.height,
                            win32gui.GetDC(win32gui.GetDesktopWindow()), 0, 0, self.width, self.height,
                            (0, 255, 255, 0))
        
        # Copy the memory DC to the window DC
        win32gui.BitBlt(self.hdc, 0, 0, self.width, self.height, self.mem_dc, 0, 0, win32con.SRCCOPY)
        
    def close(self):
        win32gui.DeleteObject(self.bitmap)
        win32gui.DeleteDC(self.mem_dc)
        win32gui.ReleaseDC(self.hwnd, self.hdc)
        win32gui.DestroyWindow(self.hwnd)


class ScreenOverlay:
    def __init__(self, args):
        self.args = args
        self.capture_width = args.width
        self.capture_height = args.height
        self.running = False
        self.upsampler = setup_upsampler(args)
        
        # Create a window for controls
        self.root = tk.Tk()
        self.root.title("RealESRGAN Screen Overlay")
        self.root.geometry("300x150")
        
        # Parse capture region
        if args.region:
            self.x, self.y, w, h = map(int, args.region.split(','))
            self.capture_width = w
            self.capture_height = h
        else:
            self.x, self.y = 0, 0
        
        # Create the overlay once user starts the filter
        self.overlay = None
        
        # Add buttons for control
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(self.button_frame, text="Start Overlay", command=self.start_overlay)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_overlay)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.stop_button.config(state=tk.DISABLED)
        
        # Add opacity slider
        self.opacity_frame = tk.Frame(self.root)
        self.opacity_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.opacity_frame, text="Opacity:").pack(side=tk.LEFT)
        self.opacity_slider = tk.Scale(self.opacity_frame, from_=0, to=100, 
                                       orient=tk.HORIZONTAL, length=200)
        self.opacity_slider.set(50)  # Default 50% opacity
        self.opacity_slider.pack(side=tk.LEFT, padx=5)
        
        # Stats
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.fps_label = tk.Label(self.stats_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.frame_count = 0
        self.last_time = time.time()
        
        # Setup close event handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Processing thread
        self.processing_thread = None

    def capture_screen(self):
        """Capture a region of the screen."""
        # Capture screen based on specified region
        screenshot = pyautogui.screenshot(region=(self.x, self.y, self.capture_width, self.capture_height))
        
        # Convert to numpy array compatible with OpenCV
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def process_frame(self, img):
        """Process a frame using RealESRGAN."""
        try:
            # Process the image
            output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
            return output
        except Exception as e:
            print(f"Error processing frame: {e}")
            return img  # Return original on error
    
    def processing_loop(self):
        """Main processing thread."""
        while self.running:
            start_time = time.time()
            
            # Capture screen
            img = self.capture_screen()
            
            # Process frame
            output = self.process_frame(img)
            
            # Update overlay
            if self.overlay:
                self.overlay.update(output)
            
            # Update FPS counter
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                # Update label in main thread
                self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
            
            # Calculate delay to maintain target frame rate
            processing_time = time.time() - start_time
            delay = max(0.001, (1.0/self.args.fps - processing_time))
            time.sleep(delay)
    
    def start_overlay(self):
        """Start the overlay process."""
        if not self.running:
            self.running = True
            # Create overlay window
            self.overlay = TransparentOverlay(self.x, self.y, self.capture_width, self.capture_height)
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
    
    def stop_overlay(self):
        """Stop the overlay process."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        if self.overlay:
            self.overlay.close()
            self.overlay = None
            
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def on_close(self):
        """Handle window close event."""
        self.stop_overlay()
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Screen Overlay")
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='realesr-animevideov3',
                        help='Model name: realesr-animevideov3, RealESRGAN_x4plus, realesr-general-x4v3')
    parser.add_argument('--outscale', type=float, default=1.0, 
                        help='The final upsampling scale of the image')
    parser.add_argument('--tile', type=int, default=0, 
                        help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, 
                        help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, 
                        help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', 
                        help='Use fp32 precision during inference. Default: fp16')
    
    # Screen capture parameters
    parser.add_argument('--width', type=int, default=640, 
                        help='Width of the capture/overlay window')
    parser.add_argument('--height', type=int, default=480, 
                        help='Height of the capture/overlay window')
    parser.add_argument('--region', type=str, default=None, 
                        help='Region to capture: x,y,width,height')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Target FPS')
    
    args = parser.parse_args()
    
    # Create and run the screen filter
    app = ScreenOverlay(args)
    app.run()


if __name__ == "__main__":
    main()