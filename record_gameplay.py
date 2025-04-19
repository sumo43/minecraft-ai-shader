import argparse
import cv2
import numpy as np
import os
import torch
import time
import threading
import keyboard
from mss import mss
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils_patched import RealESRGANer
from datetime import datetime
try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

class VideoWriter:
    def __init__(self, output_path, width, height, fps=30, outscale=1.0):
        """
        Initialize ffmpeg video writer
        """
        out_width, out_height = int(width * outscale), int(height * outscale)
        
        self.output_path = output_path
        self.closed = False
        
        try:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                            framerate=fps)
                .output(output_path, pix_fmt='yuv420p', vcodec='libx264', preset='fast')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True))
            print(f"Video writer initialized for {output_path}")
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            self.stream_writer = None
            self.closed = True

    def write_frame(self, frame):
        """Write a frame to the video file"""
        if self.closed or self.stream_writer is None:
            raise ValueError("Cannot write to closed video writer")
        
        try:
            frame = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame)
        except Exception as e:
            print(f"Error writing frame to {self.output_path}: {e}")
            self.closed = True
            raise

    def close(self):
        """Close the video writer"""
        if self.closed or self.stream_writer is None:
            print("Writer already closed")
            return
            
        try:
            self.stream_writer.stdin.close()
            self.stream_writer.wait()
            print(f"Video file {self.output_path} closed successfully")
        except Exception as e:
            print(f"Error closing video writer: {e}")
        finally:
            self.closed = True


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


class ScreenRecorder:
    def __init__(self, args):
        self.args = args
        self.capture_width = args.width
        self.capture_height = args.height
        self.running = False
        self.recording = False
        self.writer = None
        self.output_path = None
        self.current_frame = None
        
        # Settings
        if self.args.region:
            self.region = list(map(int, args.region.split(',')))
        else:
            self.region = None
        
        # Stats
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        
        # Initialize model
        print("Setting up model (this may take a few moments)...")
        self.upsampler = setup_upsampler(args)
        
        # Warmup
        print("Warming up model...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.process_frame(dummy_img)
        print("Model ready.")
        
        # Setup key event handler
        keyboard.on_press_key("f7", self.toggle_recording)
        
    def capture_screen(self):
        """Capture a region of the screen using mss."""
        try:
            with mss() as sct:
                if self.region:
                    x, y, w, h = self.region
                    monitor = {"left": x, "top": y, "width": w, "height": h}
                else:
                    # Capture the first monitor
                    monitor = sct.monitors[1]  # 1 is the first monitor (0 is all monitors combined)
                
                # Capture the screen
                sct_img = sct.grab(monitor)
                
                # Convert to numpy array
                img = np.array(sct_img)
                
                # Resize if needed
                if img.shape[0] != self.capture_height or img.shape[1] != self.capture_width:
                    img = cv2.resize(img, (self.capture_width, self.capture_height))
                
                # Convert from BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                return img
        except Exception as e:
            import traceback
            print(f"Error capturing screen: {e}")
            traceback.print_exc()
            return np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
    
    def process_frame(self, img):
        """Process a frame using RealESRGAN."""
        try:
            # Apply downscaling if specified (to improve performance)
            if self.args.downscale > 1:
                proc_h, proc_w = img.shape[0] // self.args.downscale, img.shape[1] // self.args.downscale
                img_small = cv2.resize(img, (proc_w, proc_h))
                
                # Adjust outscale to compensate for the downscaling
                effective_outscale = self.args.outscale * self.args.downscale
                
                # Process using enhanced image with effective outscale
                output, _ = self.upsampler.enhance(img_small, outscale=effective_outscale)
            else:
                # Process the image at full resolution
                output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
            
            # If output is None (e.g., due to error), use original image
            if output is None:
                print("Output is None, using original image")
                return img
                
            return output
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return img  # Return original on error
    
    def toggle_recording(self, e):
        """Toggle recording on/off when F7 is pressed"""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"gameplay_recording_{timestamp}.mp4"
            print(f"Starting recording to {self.output_path}")
            self.writer = VideoWriter(
                self.output_path, 
                self.capture_width, 
                self.capture_height, 
                fps=self.args.fps,
                outscale=self.args.outscale
            )
            self.recording = True
        else:
            # Mark recording as stopped first to prevent further frame writes
            self.recording = False
            time.sleep(0.1)  # Small delay to ensure no frames are being written
            
            # Stop recording and close file
            if self.writer is not None:
                print(f"Stopping recording, saved to {self.output_path}")
                try:
                    self.writer.close()
                except Exception as e:
                    print(f"Error closing writer: {e}")
                self.writer = None
    
    def process_loop(self):
        """Main processing loop."""
        try:
            # Calculate time for FPS control
            start_time = time.time()
            
            # Capture screen
            img = self.capture_screen()
            self.current_frame = img
            
            # Process frame
            output = self.process_frame(img)
            
            # Record frame if recording is active
            if self.recording and self.writer is not None:
                try:
                    self.writer.write_frame(output)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    # If we can't write, stop recording
                    self.recording = False
                    self.writer = None
            
            # Update FPS counter
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                self.fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                print(f"FPS: {self.fps:.1f}" + 
                      f" - {'RECORDING' if self.recording else 'NOT RECORDING'}" +
                      f" - Press F7 to {'stop' if self.recording else 'start'} recording")
            
            # Sleep to maintain target frame rate
            frame_time = time.time() - start_time
            sleep_time = max(0, (1.0 / self.args.fps) - frame_time)
            time.sleep(sleep_time)
            
        except Exception as e:
            import traceback
            print(f"Error in processing loop: {e}")
            traceback.print_exc()
    
    def run(self):
        """Run the recorder."""
        try:
            self.running = True
            print("Screen recorder running. Press F7 to start/stop recording.")
            print("Press Ctrl+C to exit")
            
            while self.running:
                self.process_loop()
                
        except KeyboardInterrupt:
            print("Exiting...")
            if self.recording and self.writer is not None:
                print(f"Stopping recording, saved to {self.output_path}")
                self.writer.close()
            self.running = False


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Screen Recorder")
    
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
    
    # Screen capture parameters
    parser.add_argument('--width', type=int, default=1280, 
                        help='Width of the capture window')
    parser.add_argument('--height', type=int, default=720, 
                        help='Height of the capture window')
    parser.add_argument('--region', type=str, default=None, 
                        help='Region to capture: x,y,width,height')
    parser.add_argument('--fps', type=int, default=5, 
                        help='Target FPS for recording (default: 5)')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    
    args = parser.parse_args()
    
    # Create and run the recorder
    recorder = ScreenRecorder(args)
    recorder.run()


if __name__ == "__main__":
    main()