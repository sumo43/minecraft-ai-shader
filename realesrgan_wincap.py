import argparse
import cv2
import numpy as np
import os
import torch
import time
import datetime
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_patched import RealESRGANer

# Global variables
upsampler = None
args = None
processed_frames = 0
start_time = time.time()

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
    
    if args.compile:
        try:
            print("Compiling model for better performance...")
            upsampler.model = torch.compile(upsampler.model, fullgraph=True, dynamic=False, mode="reduce-overhead")
            print("Model compilation complete")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")
            print("Continuing without compilation")
    
    return upsampler

def process_frame(img, downscale, outscale):
    """Process a frame using RealESRGAN."""
    global upsampler
    
    try:
        # BGRA to BGR conversion
        if img.shape[2] == 4:  # If image has alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
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
            print("Warning: Enhancement produced None output, returning original image")
            return img
            
        return output
    except Exception as e:
        import traceback
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return img  # Return original on error

def save_frame(img, prefix="frame"):
    """Save a frame to disk with timestamp."""
    try:
        # Create timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        
        # Save the image
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} frame to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None

def main():
    global upsampler, args, processed_frames, start_time
    
    parser = argparse.ArgumentParser(description="RealESRGAN Screen Enhancer using WindowsCapture")
    
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
    parser.add_argument('--compile', action='store_true',
                        help='Compile the model for better performance')
    
    # Capture parameters
    parser.add_argument('--monitor', type=int, default=1,
                        help='Monitor index to capture (1-based)')
    parser.add_argument('--region', type=str, default=None, 
                        help='Region to capture: x,y,width,height')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    parser.add_argument('--fps_target', type=int, default=0,
                       help='Target FPS (0 = unlimited)')
    parser.add_argument('--frames', type=int, default=0,
                       help='Number of frames to process (0 = unlimited)')
    parser.add_argument('--save_originals', action='store_true',
                       help='Save original frames to disk')
    parser.add_argument('--save_enhanced', action='store_true',
                       help='Save enhanced frames to disk')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save every N frames (default=1, save all frames)')
    
    args = parser.parse_args()
    
    # Parse region if specified
    region = None
    if args.region:
        try:
            region = list(map(int, args.region.split(',')))
            if len(region) != 4:
                print("Error: Region must be specified as: x,y,width,height")
                return
        except ValueError:
            print("Error: Invalid region format. Use: x,y,width,height")
            return
    
    # Initialize model
    print("Initializing model...")
    upsampler = setup_upsampler(args)
    print("Model initialization complete")
    
    # Warmup
    print("Warming up model...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = process_frame(dummy_img, args.downscale, args.outscale)
    print("Warmup complete")
    
    # Initialize capture
    running = True
    processed_frames = 0
    start_time = time.time()
    
    print(f"Initializing capture for monitor {args.monitor}...")
    
    capture = WindowsCapture(
        cursor_capture=True,
        draw_border=False,
        monitor_index=args.monitor,
        window_name=None
    )
    
    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        nonlocal running
        global processed_frames, upsampler, args
        
        if not running:
            capture_control.stop()
            return
        
        try:
            frame_start_time = time.time()
            
            # Get the frame data from frame_buffer
            img = frame.frame_buffer
            
            # Crop if region specified
            if region:
                x, y, w, h = region
                if x + w <= img.shape[1] and y + h <= img.shape[0]:
                    img = img[y:y+h, x:x+w]
                else:
                    print(f"Region {region} is outside of capture bounds {img.shape[1]}x{img.shape[0]}")
            
            # Save original frame if requested
            if args.save_originals and processed_frames % args.save_every == 0:
                save_frame(img, f"original_{processed_frames}")
            
            # Process the image
            enhanced_img = process_frame(img, args.downscale, args.outscale)
            
            # Save enhanced frame if requested
            if args.save_enhanced and processed_frames % args.save_every == 0:
                save_frame(enhanced_img, f"enhanced_{processed_frames}")
            
            # Update statistics
            processed_frames += 1
            frame_time = time.time() - frame_start_time
            elapsed = time.time() - start_time
            fps = processed_frames / elapsed
            
            # Print statistics every second
            if int(elapsed) > int(elapsed - frame_time):
                print(f"Frame {processed_frames}: {frame_time*1000:.1f}ms ({fps:.1f} FPS)")
            
            # Enforce target FPS if specified
            if args.fps_target > 0:
                target_frame_time = 1.0 / args.fps_target
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
            
            # Check if we should stop
            if args.frames > 0 and processed_frames >= args.frames:
                print(f"Processed {args.frames} frames, stopping")
                running = False
                capture_control.stop()
            
        except Exception as e:
            import traceback
            print(f"Error in frame processing: {e}")
            traceback.print_exc()
    
    @capture.event
    def on_closed():
        nonlocal running
        print("Capture session closed")
        running = False
    
    # Start capture
    print("Starting capture...")
    capture.start()
    
    # Wait for completion
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        running = False
    
    # Print final statistics
    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0
    print(f"Processed {processed_frames} frames in {total_time:.2f}s ({avg_fps:.2f} FPS)")


if __name__ == "__main__":
    main()