import argparse
import cv2
import numpy as np
import os
import torch
import time
import threading
from PIL import Image
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_patched import RealESRGANer

# Global variables
running = False
upsampler = None
args = None
processed_frames = 0
start_time = time.time()
monitor_index = 1  # Default to primary monitor (Windows Capture uses 1-based indexing)

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

def process_frame(img, downscale, outscale):
    """Process a frame using RealESRGAN."""
    global upsampler
    try:
        # Apply downscaling if specified (to improve performance)
        if downscale > 1:
            proc_h, proc_w = img.shape[0] // downscale, img.shape[1] // downscale
            img_small = cv2.resize(img, (proc_w, proc_h))
            
            # Adjust outscale to compensate for the downscaling
            effective_outscale = outscale * downscale
            
            # Process using enhanced image with effective outscale
            output, _ = upsampler.enhance(img_small, outscale=effective_outscale)
        else:
            # Process the image at full resolution
            output, _ = upsampler.enhance(img, outscale=outscale)
        
        if output is None:
            print("Output is None, returning original image")
            return img
            
        return output
    except Exception as e:
        import traceback
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return img  # Return original on error

def save_frame(img, prefix="original"):
    """Save a frame to disk with timestamp."""
    try:
        # Create timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        
        # Save the image
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} frame to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None

def start_capture():
    """Start screen capture using WindowsCapture."""
    global running, upsampler, args, processed_frames, start_time, monitor_index
    
    # Reset counters
    processed_frames = 0
    start_time = time.time()
    
    print("Starting screen capture...")
    
    # Initialize WindowsCapture
    if args.region:
        # We'll handle this in the frame processing since WindowsCapture 
        # doesn't seem to support direct region capture
        use_monitor = True
    else:
        use_monitor = True
    
    # Create capture instance with appropriate settings
    capture = WindowsCapture(
        cursor_capture=True,
        draw_border=False,
        monitor_index=monitor_index if use_monitor else None,
        window_name=None  # We'll capture the entire monitor or specified region
    )
    
    # Set up event handlers
    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        global running, processed_frames, upsampler, args
        
        if not running:
            capture_control.stop()
            return
        
        try:
            # Get the image data from frame_buffer
            img = frame.frame_buffer
            print(f"Captured frame with shape: {img.shape}, dtype: {img.dtype}")
            
            # If a region is specified, crop the image
            if args.region:
                x, y, w, h = args.region
                if x + w <= img.shape[1] and y + h <= img.shape[0]:
                    img = img[y:y+h, x:x+w]
                    print(f"Cropped to region: {args.region}, new shape: {img.shape}")
                else:
                    print(f"Region {args.region} is outside of capture bounds {img.shape[1]}x{img.shape[0]}")
            
            # Always save first frame
            save_frame(img, "capture_test")
            print("Saved first frame, stopping capture")
            
            # For testing only - just capture one frame and exit
            running = False
            capture_control.stop()
            return
            
            # Process the image (only reached if not in test mode)
            output = process_frame(img, args.downscale, args.outscale)
            
            # Save frames if requested
            if args.save_frames:
                save_frame(img, "original")
                save_frame(output, "enhanced")
            
            # Update statistics
            processed_frames += 1
            elapsed = time.time() - start_time
            if processed_frames % 30 == 0:  # Show stats every 30 frames
                fps = processed_frames / elapsed
                print(f"Processed {processed_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
            
        except Exception as e:
            import traceback
            print(f"Error in frame processing: {e}")
            traceback.print_exc()
    
    @capture.event
    def on_closed():
        global running
        print("Capture session closed")
        running = False
    
    # Start capture
    running = True
    capture.start()

def main():
    global upsampler, args, monitor_index, running
    
    parser = argparse.ArgumentParser(description="RealESRGAN Screen Filter with WindowsCapture")
    
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
    parser.add_argument('--fast_mode', action='store_true',
                        help='Use optimized model for faster processing')
    
    # Capture parameters
    parser.add_argument('--region', type=str, default=None, 
                        help='Region to capture: x,y,width,height')
    parser.add_argument('--monitor', type=int, default=1,
                        help='Monitor index to capture (1-based)')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save frames to disk during processing')
    
    args = parser.parse_args()
    
    # Parse region if specified
    if args.region:
        try:
            args.region = list(map(int, args.region.split(',')))
            if len(args.region) != 4:
                print("Error: Region must be specified as: x,y,width,height")
                return
        except ValueError:
            print("Error: Invalid region format. Use: x,y,width,height")
            return
    
    # Set monitor index
    monitor_index = args.monitor
    
    print("Initializing model...")
    upsampler = setup_upsampler(args)
    print("Model initialized")
    
    # Warm up model
    print("Warming up model...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    if args.downscale > 1:
        dummy_img = cv2.resize(dummy_img, (640//args.downscale, 480//args.downscale))
        effective_outscale = args.outscale * args.downscale
        _ = process_frame(dummy_img, args.downscale, args.outscale)
    else:
        _ = process_frame(dummy_img, 1, args.outscale)
    print("Warmup complete")
    
    # Start capture
    start_capture()
    
    # Keep main thread alive
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        running = False

if __name__ == "__main__":
    main()