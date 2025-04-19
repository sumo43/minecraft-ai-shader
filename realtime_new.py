import argparse
import cv2
import numpy as np
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_patched import RealESRGANer

# Import the capture function
from cap import capture_win_alt

class RealESRGANApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.upsampler = None
        self.running = False
        self.processed_frames = 0
        self.start_time = time.time()

        # Configure the main window
        self.root.title("RealESRGAN Window Filter")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Set up the UI
        self.setup_ui()

        # Initialize model in background thread
        self.status_var.set("Initializing model...")
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def setup_ui(self):
        """Set up the user interface"""
        # Status bar
        self.status_var = tk.StringVar()
        
        if self.args.overlay_mode:
            # Overlay mode - fullscreen window with just the enhanced output
            # Configure fullscreen
            self.root.attributes('-fullscreen', True)
            self.root.overrideredirect(True)  # Removes window borders
            
            # Create a frame to hold everything
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create a canvas that fills the screen for the enhanced output
            self.canvas_enhanced = tk.Canvas(main_frame, bg='black')
            self.canvas_enhanced.pack(fill=tk.BOTH, expand=True)
            
            # Add a small status bar at the bottom
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Add key bindings for exiting overlay mode
            self.root.bind('<Escape>', lambda e: self.on_close())
            self.root.bind('q', lambda e: self.on_close())
            
            # Start automatically in overlay mode
            self.root.after(100, self.start_capture)
            
            # Set variables needed for normal mode (even if we don't use them)
            self.canvas_orig = None
            self.downscale_var = tk.IntVar(value=self.args.downscale)
            self.outscale_var = tk.DoubleVar(value=self.args.outscale)
            self.fps_var = tk.StringVar(value="FPS: 0.0")
        else:
            # Normal mode with side-by-side comparison
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10 10 10 10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            
            # Status bar
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            # Image display area - two canvases side by side
            self.display_frame = ttk.Frame(main_frame)
            self.display_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(0, weight=1)
            
            # Original image canvas
            self.canvas_orig = tk.Canvas(self.display_frame,
                                        width=self.args.width,
                                        height=self.args.height,
                                        bg='black')
            self.canvas_orig.grid(row=0, column=0, padx=5, pady=5)
            
            # Enhanced image canvas
            self.canvas_enhanced = tk.Canvas(self.display_frame,
                                            width=self.args.width,
                                            height=self.args.height,
                                            bg='black')
            self.canvas_enhanced.grid(row=0, column=1, padx=5, pady=5)
            
            # Canvas labels
            ttk.Label(self.display_frame, text="Original").grid(row=1, column=0)
            ttk.Label(self.display_frame, text="Enhanced").grid(row=1, column=1)
            
            # Control panel
            control_frame = ttk.Frame(main_frame)
            control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
            
            # Start button
            self.start_button = ttk.Button(control_frame, text="Start", command=self.start_capture, state=tk.DISABLED)
            self.start_button.grid(row=0, column=0, padx=5)
            
            # Stop button
            self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
            self.stop_button.grid(row=0, column=1, padx=5)
            
            # Placeholder for layout consistency
            ttk.Label(control_frame, text="").grid(row=0, column=2, padx=5)
            
            # FPS display
            self.fps_var = tk.StringVar(value="FPS: 0.0")
            ttk.Label(control_frame, textvariable=self.fps_var).grid(row=0, column=3, padx=20)
            
            # Downscale option
            self.downscale_var = tk.IntVar(value=self.args.downscale)
            ttk.Label(control_frame, text="Downscale:").grid(row=0, column=4, padx=(20,5))
            downscale_spin = ttk.Spinbox(control_frame, from_=1, to=4, width=5, textvariable=self.downscale_var)
            downscale_spin.grid(row=0, column=5)
            
            # Scale option
            self.outscale_var = tk.DoubleVar(value=self.args.outscale)
            ttk.Label(control_frame, text="Outscale:").grid(row=0, column=6, padx=(20,5))
            outscale_spin = ttk.Spinbox(control_frame, from_=0.5, to=4, increment=0.5, width=5, textvariable=self.outscale_var)
            outscale_spin.grid(row=0, column=7)
        
        # Current images
        self.current_original = None
        self.current_enhanced = None

    def initialize_model(self):
        """Initialize the RealESRGAN model in a background thread."""
        try:
            from basicsr.utils.download_util import load_file_from_url

            # Update status
            self.update_status("Setting up model...")

            # ---------------------- determine models according to model names ---------------------- #
            model_name = self.args.model_name.split('.pth')[0]
            """
            if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            else:
            """
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
            # Default to animevideov3 model
            #model_name = 'realesr-animevideov3'

            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']


            # ---------------------- determine model paths ---------------------- #
            model_path = os.path.join('weights', model_name + '.pth')

            print(f"model_path is {model_path}")
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(os.path.join(ROOT_DIR, 'weights'), exist_ok=True)
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

            # Model compilation happens in RealESRGANer class initialization
            # Check if torch.compile is available
            if hasattr(torch, 'compile'):
                self.update_status("Setting up model with torch.compile optimization...")
            else:
                self.update_status("torch.compile not available (requires PyTorch 2.0+)")

            print(f"loading model from weights/{model_path}")
            # Initialize the upsampler
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=self.args.tile,
                tile_pad=self.args.tile_pad,
                pre_pad=self.args.pre_pad,
                half=not self.args.fp32,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            )

            # Warmup
            self.update_status("Warming up model...")
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.process_frame(dummy_img, self.args.downscale, self.args.outscale)

            # Enable UI
            self.update_status("Ready")
            self.root.after(0, self.enable_ui)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"Error initializing model: {str(e)}")

    def update_status(self, message):
        """Update status bar message."""
        if self.args.overlay_mode and self.running:
            # In overlay mode and running, preserve FPS info
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            self.root.after(0, lambda: self.status_var.set(f"{message} | FPS: {fps:.1f} | Press ESC or 'q' to exit"))
        else:
            self.root.after(0, lambda: self.status_var.set(message))

    def enable_ui(self):
        """Enable UI elements after initialization."""
        if not self.args.overlay_mode:
            self.start_button.config(state=tk.NORMAL)

    def capture_window(self):
        """Capture the target window using capture_win_alt."""
        try:
            # Capture the window specified by window_name
            window_name = self.args.window_name
            img = capture_win_alt(convert=True, window_name=window_name)
            
            # Resize if needed
            if img.shape[1] != self.args.width or img.shape[0] != self.args.height:
                img = cv2.resize(img, (self.args.width, self.args.height))
            
            return img
        except Exception as e:
            print(f"Error capturing window: {e}")
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

    def process_frame(self, img, downscale, outscale):
        """Process a frame using RealESRGAN."""
        try:
            # RGB to BGR conversion if needed
            # The capture function returns RGB array from PIL, so we need to convert
            if img.dtype == np.uint8 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img  # Assume already in the correct format

            # Apply downscaling if specified (to improve performance)
            if downscale > 1:
                proc_h, proc_w = img_bgr.shape[0] // downscale, img_bgr.shape[1] // downscale
                img_small = cv2.resize(img_bgr, (proc_w, proc_h))

                # Adjust outscale to compensate for the downscaling
                effective_outscale = outscale * downscale

                # Process using enhanced image with effective outscale
                output, _ = self.upsampler.enhance(img_small, outscale=effective_outscale)

            else:
                # Process the image at full resolution
                output, _ = self.upsampler.enhance(img_bgr, outscale=outscale)

            if output is None:
                print("Warning: Enhancement produced None output, returning original image")
                return img_bgr

            return output
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            if img.dtype == np.uint8 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Return original in BGR
            return img

    def update_display(self, original_img, enhanced_img):
        """Update the display with original and enhanced images."""
        try:
            # Store current images
            self.current_original = original_img
            self.current_enhanced = enhanced_img

            # Convert enhanced image from BGR to RGB
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

            # Create PIL image for enhanced view
            pil_enhanced = Image.fromarray(enhanced_rgb)
            
            if self.args.overlay_mode:
                # In overlay mode, resize the enhanced image to fill the screen
                screen_width = self.root.winfo_width()
                screen_height = self.root.winfo_height()
                
                if screen_width <= 1 or screen_height <= 1:
                    # Window hasn't been drawn yet, use the specified dimensions
                    screen_width = 1920  # Default to standard HD width
                    screen_height = 1080  # Default to standard HD height
                
                # Resize while maintaining aspect ratio and filling the screen
                img_ratio = pil_enhanced.width / pil_enhanced.height
                screen_ratio = screen_width / screen_height
                
                if img_ratio > screen_ratio:
                    # Image is wider than screen ratio
                    new_height = screen_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller than screen ratio
                    new_width = screen_width
                    new_height = int(new_width / img_ratio)
                
                pil_enhanced = pil_enhanced.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to Tkinter PhotoImage
                self.tk_enhanced = ImageTk.PhotoImage(image=pil_enhanced)
                
                # Calculate position to center the image
                x_pos = (screen_width - new_width) // 2
                y_pos = (screen_height - new_height) // 2
                
                # Clear the canvas and update with new image
                self.canvas_enhanced.delete("all")
                self.canvas_enhanced.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_enhanced)
                
                # Update the status bar with performance info
                elapsed = time.time() - self.start_time
                fps = self.processed_frames / elapsed if elapsed > 0 else 0
                self.status_var.set(f"Capturing {self.args.window_name} | FPS: {fps:.1f} | Press ESC or 'q' to exit")
            else:
                # Normal mode with side-by-side comparison
                # Create PIL image for original view
                pil_original = Image.fromarray(original_img)
                
                # Convert to Tkinter PhotoImage
                self.tk_original = ImageTk.PhotoImage(image=pil_original)
                self.tk_enhanced = ImageTk.PhotoImage(image=pil_enhanced)

                # Update canvases
                self.canvas_orig.create_image(0, 0, anchor=tk.NW, image=self.tk_original)
                self.canvas_enhanced.create_image(0, 0, anchor=tk.NW, image=self.tk_enhanced)
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()

    def capture_loop(self):
        """Main capture and processing loop."""
        if not self.running:
            return

        try:
            frame_start_time = time.time()

            # Capture window
            original_img = self.capture_window()

            # Get current settings from UI
            downscale = self.downscale_var.get()
            outscale = self.outscale_var.get()

            # Process the image
            enhanced_img = self.process_frame(original_img, downscale, outscale)

            # Update display
            self.update_display(original_img, enhanced_img)

            # Update statistics
            self.processed_frames += 1
            frame_time = time.time() - frame_start_time
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0

            # Update FPS display every 5 frames
            if self.processed_frames % 5 == 0 and not self.args.overlay_mode:
                self.fps_var.set(f"FPS: {fps:.1f}")

            # Calculate delay to maintain target FPS
            if self.args.fps_target > 0:
                target_frame_time = 1.0 / self.args.fps_target
                delay = int(max(1, (target_frame_time - frame_time) * 1000))
            else:
                delay = 1  # Default to 1ms delay

            # Schedule next frame
            self.root.after(delay, self.capture_loop)

        except Exception as e:
            import traceback
            print(f"Error in capture loop: {e}")
            traceback.print_exc()
            # Try to recover
            self.root.after(1000, self.capture_loop)

    def start_capture(self):
        """Start the capture process."""
        if not self.running:
            self.running = True
            self.processed_frames = 0
            self.start_time = time.time()
            if not self.args.overlay_mode:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
            self.update_status(f"Capturing {self.args.window_name}...")
            self.capture_loop()

    def stop_capture(self):
        """Stop the capture process."""
        self.running = False
        if not self.args.overlay_mode:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        self.update_status("Stopped")

    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Window Enhancer with Live Preview")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='realesr-animevideov3',
                        help='Model name: realesr-animevideov3, RealESRGAN_x4plus')
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
    parser.add_argument('--fast_mode', action='store_true',
                        help='Use optimized model for faster processing')

    # Capture parameters
    parser.add_argument('--width', type=int, default=640,
                        help='Width of the display window')
    parser.add_argument('--height', type=int, default=480,
                        help='Height of the display window')
    parser.add_argument('--window_name', type=str, default="Minecraft",
                        help='Title of the window to capture')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    parser.add_argument('--fps_target', type=int, default=0,
                       help='Target FPS (0 = unlimited)')
    parser.add_argument('--dummy_frame', type=str, default=None,
                       help='Path to a JPEG image to use instead of window capture')
    parser.add_argument('--overlay_mode', action='store_true',
                       help='Run in overlay mode (fullscreen with enhanced output only)')

    args = parser.parse_args()

    # Create and run the application
    root = tk.Tk()
    app = RealESRGANApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()