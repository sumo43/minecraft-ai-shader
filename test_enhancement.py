import argparse
import cv2
import numpy as np
import os
import torch
import time
import tkinter as tk
from PIL import Image, ImageTk
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
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
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

    # Set up model for real-time processing
    if args.fast_mode:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, act_type='prelu')
        print(f"Using fast mode with optimized model")

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
    
    return upsampler


class TestEnhancement:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.running = False
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Test Enhancement")
        self.root.geometry(f"{self.width*2}x{self.height}")
        
        # Create two canvases side by side
        self.canvas_orig = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas_orig.pack(side=tk.LEFT)
        
        self.canvas_enhanced = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas_enhanced.pack(side=tk.RIGHT)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Initializing model...", font=("Arial", 14))
        self.status_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Initialize model in background
        self.initialize_model()
        
        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def initialize_model(self):
        try:
            print("Setting up model...")
            self.upsampler = setup_upsampler(self.args)
            print("Model setup complete")
            
            # Create and process test image once model is ready
            self.status_label.config(text="Processing test image...")
            self.process_test_image()
            
            # Hide status label
            self.status_label.place_forget()
        except Exception as e:
            import traceback
            print(f"Error initializing model: {e}")
            traceback.print_exc()
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
    
    def create_test_image(self):
        # Create a test image (colorful gradient)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create a gradient
        for i in range(self.height):
            for j in range(self.width):
                img[i, j, 0] = i * 255 // self.height  # Blue gradient
                img[i, j, 1] = j * 255 // self.width  # Green gradient
                img[i, j, 2] = ((i + j) * 255) // (self.height + self.width)  # Red gradient
        
        return img
    
    def process_test_image(self):
        # Create test image
        img = self.create_test_image()
        
        # Display original image
        self.display_image(img, self.canvas_orig)
        
        try:
            print("Enhancing image...")
            # Process with downscaling if specified
            if self.args.downscale > 1:
                proc_h, proc_w = img.shape[0] // self.args.downscale, img.shape[1] // self.args.downscale
                img_small = cv2.resize(img, (proc_w, proc_h))
                print(f"Downscaled to: {img_small.shape}")
                
                # Adjust outscale to compensate for the downscaling
                effective_outscale = self.args.outscale * self.args.downscale
                print(f"Effective outscale: {effective_outscale}")
                
                # Process using enhancer
                enhanced, _ = self.upsampler.enhance(img_small, outscale=effective_outscale)
            else:
                # Process at full resolution
                print("Processing at full resolution")
                enhanced, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
            
            print(f"Enhancement complete, result shape: {enhanced.shape}")
            
            # Display enhanced image
            self.display_image(enhanced, self.canvas_enhanced)
            
        except Exception as e:
            import traceback
            print(f"Error enhancing image: {e}")
            traceback.print_exc()
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            self.status_label.place(relx=0.75, rely=0.5, anchor=tk.CENTER)
    
    def display_image(self, img, canvas):
        # Convert to PIL format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Display the image
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk  # Keep a reference
    
    def on_close(self):
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Test Enhancement")
    
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
    
    # Image parameters
    parser.add_argument('--width', type=int, default=640, 
                       help='Width of the test image')
    parser.add_argument('--height', type=int, default=480, 
                       help='Height of the test image')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    
    args = parser.parse_args()
    
    app = TestEnhancement(args)
    app.run()

if __name__ == "__main__":
    main()