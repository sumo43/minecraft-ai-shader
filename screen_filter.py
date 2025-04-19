import argparse
import cv2
import numpy as np
import os
import torch
import time
import threading
import pyautogui
from PIL import Image, ImageTk
import tkinter as tk
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_patched import RealESRGANer


def setup_upsampler(args):
    """Set up the RealESRGAN upsampler."""
    from basicsr.utils.download_util import load_file_from_url

    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        # Default to animevideov3 model
        args.model_name = 'realesr-animevideov3'
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(ROOT_DIR, 'weights'), exist_ok=True)
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength for realesr-general-x4v3
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # Set up the model for real-time processing (preferring smaller model)
    if args.fast_mode:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, act_type='prelu')
        print(f"Using fast mode with optimized model, params: {sum(p.numel() for p in model.parameters())}")

    # Initialize the upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # Compile the model for better performance
    #upsampler.model = torch.compile(upsampler.model, fullgraph=True, dynamic=False, mode="reduce-overhead")

    return upsampler


class ScreenFilter:
    def __init__(self, args):
        self.args = args
        self.capture_width = args.width
        self.capture_height = args.height
        self.running = False

        # Create a window to display the upscaled content
        self.root = tk.Tk()
        self.root.title("RealESRGAN Screen Filter")
        self.root.geometry(f"{self.capture_width}x{self.capture_height}")

        # Canvas for displaying the processed image
        self.canvas = tk.Canvas(self.root, width=self.capture_width, height=self.capture_height)
        self.canvas.pack()

        # Status label for initialization
        self.status_label = tk.Label(self.root, text="Initializing model...", font=("Arial", 14))
        self.status_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Add buttons for control
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_filter, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_filter, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Setup close event handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Stats
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_label = tk.Label(self.button_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Initialize model in a background thread to keep UI responsive
        self.init_thread = threading.Thread(target=self.initialize_model)
        self.init_thread.daemon = True
        self.init_thread.start()

    def initialize_model(self):
        """Initialize the model in background and perform warmup."""
        try:
            # Setup upsampler
            self.upsampler = setup_upsampler(self.args)

            # Create dummy input for warmup (compile the model)
            if self.args.warmup:
                self.root.after(0, lambda: self.status_label.config(text="Warming up model (may take a few seconds)..."))
                dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Run inference once to compile the model
                _ = self.process_frame(dummy_img)

            # Enable start button
            self.root.after(0, lambda: self.status_label.place_forget())
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error initializing model: {str(e)}",
                fg="red"
            ))

    def capture_screen(self):
        """Capture a region of the screen."""
        # Capture screen based on specified region
        if self.args.region:
            x, y, w, h = map(int, self.args.region.split(','))
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
        else:
            # Capture entire screen and resize
            screenshot = pyautogui.screenshot()
            screenshot = screenshot.resize((self.capture_width, self.capture_height))

        # Convert to numpy array compatible with OpenCV
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def process_frame(self, img):
        """Process a frame using RealESRGAN."""
        try:
            # Ensure the image is the correct shape and type
            if img.shape[0] != self.capture_height or img.shape[1] != self.capture_width:
                img = cv2.resize(img, (self.capture_width, self.capture_height))

            print(f"Processing frame shape: {img.shape}")

            # Apply downscaling if specified (to improve performance)
            if self.args.downscale > 1:
                proc_h, proc_w = img.shape[0] // self.args.downscale, img.shape[1] // self.args.downscale
                img_small = cv2.resize(img, (proc_w, proc_h))
                print(f"Downscaled to: {img_small.shape}")

                # Adjust outscale to compensate for the downscaling
                effective_outscale = self.args.outscale * self.args.downscale
                print(f"Effective outscale: {effective_outscale}")

                # Process using enhanced image with effective outscale
                output, _ = self.upsampler.enhance(img_small, outscale=effective_outscale)
            else:
                # Process the image at full resolution
                print("Processing at full resolution")
                output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)

            print(f"Output shape: {output.shape if output is not None else 'None'}")

            # If output is None (e.g., due to error), use original image
            if output is None:
                print("Output is None, using original image")
                return img

            # Resize if necessary to fit display window
            if output.shape[0] != self.capture_height or output.shape[1] != self.capture_width:
                output = cv2.resize(output, (self.capture_width, self.capture_height))

            return output
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return img  # Return original on error

    def update_display(self, img):
        """Update the TKinter canvas with the processed image."""
        try:
            print(f"Updating display with image of shape: {img.shape}")
            print(f"Image min: {img.min()}, max: {img.max()}, type: {img.dtype}")
            
            # Convert image format for Tkinter
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("After cv2.cvtColor")
            
            img_pil = Image.fromarray(img_rgb)
            print("After Image.fromarray")
            
            img_tk = ImageTk.PhotoImage(image=img_pil)
            print("After ImageTk.PhotoImage")
            
            # Update canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            print("After canvas.create_image")
            
            self.canvas.image = img_tk  # Keep a reference
            print("Display update complete")
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()

        # Update FPS counter
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            current_time = time.time()
            fps = 10 / (current_time - self.last_time)
            self.last_time = current_time
            self.fps_label.config(text=f"FPS: {fps:.1f}")

    def filter_loop(self):
        """Main processing loop."""
        if not self.running:
            return
        
        try:
            print("\n--- Starting new frame ---")
            # Capture and process
            start_time = time.time()
            
            # Capture screen
            img = self.capture_screen()
            print(f"Screen capture complete, image shape: {img.shape}")
            
            # Process frame
            print("Processing frame...")
            output = self.process_frame(img)
            print(f"Processing complete, output shape: {output.shape if output is not None else 'None'}")
            
            # Update display
            print("Updating display...")
            self.update_display(output)
            
            # Calculate delay to maintain target frame rate
            processing_time = time.time() - start_time
            print(f"Frame processing time: {processing_time:.3f}s")
            delay = max(1, int((1.0/self.args.fps - processing_time) * 1000))
            print(f"Scheduling next frame with delay: {delay}ms")
            
            # Schedule next frame
            self.root.after(delay, self.filter_loop)
        except Exception as e:
            import traceback
            print(f"Error in filter loop: {e}")
            traceback.print_exc()
            
            # Try to continue with next frame
            self.root.after(1000, self.filter_loop)

    def start_filter(self):
        """Start the filtering process."""
        if not self.running:
            self.running = True
            self.filter_loop()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

    def stop_filter(self):
        """Stop the filtering process."""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Screen Filter")

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
    parser.add_argument('--denoise_strength', type=float, default=0.5,
                        help='Denoise strength for realesr-general-x4v3 model')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Use optimized model for faster processing')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='Perform model warmup to avoid first-frame lag')

    # Screen capture parameters
    parser.add_argument('--width', type=int, default=1280,
                        help='Width of the capture/display window')
    parser.add_argument('--height', type=int, default=720,
                        help='Height of the capture/display window')
    parser.add_argument('--region', type=str, default=None,
                        help='Region to capture: x,y,width,height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')

    args = parser.parse_args()

    # Create and run the screen filter
    app = ScreenFilter(args)
    app.run()


if __name__ == "__main__":
    main()