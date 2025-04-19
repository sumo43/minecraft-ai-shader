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

class VideoOverlayApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.upsampler = None
        self.running = False
        self.processed_frames = 0
        self.start_time = time.time()
        
        # Configure the main window for overlay mode
        self.root.title("RealESRGAN Video Overlay")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.attributes('-fullscreen', True)
        self.root.overrideredirect(True)  # Removes window borders
        self.root.attributes('-topmost', True)  # Keep window on top
        
        # Make window transparent if supported
        try:
            self.root.attributes('-transparentcolor', 'black')
            self.transparent_bg = True
        except:
            self.transparent_bg = False
            print("Transparent window not supported on this platform")
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize model in background thread
        self.status_var.set("Initializing model...")
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def setup_ui(self):
        """Set up the overlay user interface"""
        # Status bar
        self.status_var = tk.StringVar()
        
        # Create a frame to hold everything
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas that fills the screen for the enhanced output
        self.canvas = tk.Canvas(main_frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add a small status bar at the bottom
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add key bindings
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('q', lambda e: self.on_close())
        self.root.bind('t', lambda e: self.toggle_transparency())
        self.root.bind('<space>', lambda e: self.toggle_pause())
        
        # Variables for processing
        self.downscale_var = self.args.downscale
        self.outscale_var = self.args.outscale
        
        # Current image
        self.current_enhanced = None
        
        # Playback state
        self.paused = False

    def toggle_transparency(self):
        """Toggle background transparency"""
        if not self.transparent_bg:
            return
            
        if self.canvas.cget('bg') == 'black':
            self.canvas.config(bg='systemTransparent')
            self.update_status("Transparent mode ON")
        else:
            self.canvas.config(bg='black')
            self.update_status("Transparent mode OFF")
            
    def toggle_pause(self):
        """Pause/resume playback"""
        self.paused = not self.paused
        if self.paused:
            self.update_status("Paused")
        else:
            self.update_status("Playing")

    def initialize_model(self):
        """Initialize the RealESRGAN model in a background thread."""
        try:
            from basicsr.utils.download_util import load_file_from_url

            # Update status
            self.update_status("Setting up model...")

            # ---------------------- determine models according to model names ---------------------- #
            model_name = self.args.model_name.split('.pth')[0]
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
            
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

            # Model compilation happens in RealESRGANer
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

            # Setup successful, start processing frames
            self.update_status("Ready | Press 't' to toggle transparency | SPACE to pause | ESC to exit")
            self.root.after(100, self.start_processing)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"Error initializing model: {str(e)}")

    def update_status(self, message):
        """Update status bar message."""
        if self.running:
            # Show FPS when running
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            self.root.after(0, lambda: self.status_var.set(f"{message} | FPS: {fps:.1f} | t=transparency | SPACE=pause | ESC=exit"))
        else:
            self.root.after(0, lambda: self.status_var.set(message))

    def process_frame(self, img, downscale, outscale):
        """Process a frame using RealESRGAN."""
        try:
            # Make sure image is in BGR format for the model
            if len(img.shape) == 3 and img.shape[2] == 3:
                # If RGB, convert to BGR
                if img.dtype == np.uint8:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img  # Assume already BGR
            else:
                print("Warning: Unexpected image format")
                img_bgr = img

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
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.dtype == np.uint8 else img

            return output
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.dtype == np.uint8 else img

    def update_display(self, enhanced_img):
        """Update the overlay display with the enhanced image."""
        try:
            # Store current image
            self.current_enhanced = enhanced_img

            # Convert enhanced image from BGR to RGB for PIL
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

            # Create PIL image
            pil_enhanced = Image.fromarray(enhanced_rgb)
            
            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Resize while maintaining aspect ratio to fill screen
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
            self.canvas.delete("all")
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_enhanced)
            
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()

    def load_frame_from_file(self):
        """Load the next frame from the input video file."""
        if not hasattr(self, 'cap') or self.cap is None:
            # Open the video file
            self.cap = cv2.VideoCapture(self.args.input_file)
            if not self.cap.isOpened():
                print(f"Error: Could not open video file: {self.args.input_file}")
                return None
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Opened video with {self.total_frames} frames at {self.fps} FPS")
            
        # Skip if paused
        if self.paused:
            return self.last_frame if hasattr(self, 'last_frame') else None
            
        # Read the next frame
        ret, frame = self.cap.read()
        if not ret:
            # End of video, loop if specified
            if self.args.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None
                
        # Save frame for pause mode
        self.last_frame = frame
                
        # Resize if needed
        if self.args.width > 0 and self.args.height > 0:
            if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
                frame = cv2.resize(frame, (self.args.width, self.args.height))
                
        return frame

    def process_loop(self):
        """Main processing loop."""
        if not self.running:
            return

        try:
            frame_start_time = time.time()

            # Load the next frame from file
            frame = self.load_frame_from_file()
            if frame is None:
                if not self.paused:
                    print("End of video file reached")
                    self.stop_processing()
                return

            # Process the frame
            enhanced_img = self.process_frame(frame, self.downscale_var, self.outscale_var)

            # Update the overlay display
            self.update_display(enhanced_img)

            # Update statistics
            self.processed_frames += 1
            frame_time = time.time() - frame_start_time
            
            # Calculate delay to maintain source FPS
            if self.args.fps_target > 0:
                target_fps = self.args.fps_target
            else:
                target_fps = self.fps if hasattr(self, 'fps') else 30
                
            target_frame_time = 1.0 / target_fps
            delay = int(max(1, (target_frame_time - frame_time) * 1000))
            
            # Update status occasionally
            if self.processed_frames % 5 == 0:
                if self.paused:
                    self.update_status("Paused")
                else:
                    self.update_status("Playing")

            # Schedule next frame
            self.root.after(delay, self.process_loop)

        except Exception as e:
            import traceback
            print(f"Error in process loop: {e}")
            traceback.print_exc()
            # Try to recover
            self.root.after(1000, self.process_loop)

    def start_processing(self):
        """Start the processing loop."""
        if not self.running:
            self.running = True
            self.processed_frames = 0
            self.start_time = time.time()
            self.update_status("Processing started...")
            self.process_loop()

    def stop_processing(self):
        """Stop the processing loop."""
        self.running = False
        self.update_status("Stopped")
        
        # Close video capture if active
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        """Handle window close event."""
        self.stop_processing()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Video Overlay")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='realesr-animevideov3',
                        help='Model name: realesr-animevideov3, RealESRGAN_x4plus')
    parser.add_argument('--outscale', type=float, default=2.0,
                        help='The final upsampling scale of the image')
    parser.add_argument('--tile', type=int, default=0,
                        help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10,
                        help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0,
                        help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true',
                        help='Use fp32 precision during inference. Default: fp16')

    # Input parameters
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--width', type=int, default=0,
                       help='Width to resize input frames (0 for no resize)')
    parser.add_argument('--height', type=int, default=0,
                       help='Height to resize input frames (0 for no resize)')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    parser.add_argument('--fps_target', type=int, default=0,
                       help='Target FPS (0 = match source video)')
    parser.add_argument('--loop', action='store_true',
                       help='Loop the video when it reaches the end')

    args = parser.parse_args()

    # Create and run the overlay application
    root = tk.Tk()
    app = VideoOverlayApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()