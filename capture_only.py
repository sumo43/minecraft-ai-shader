import argparse
import cv2
import numpy as np
import datetime
import os
from mss import mss

def capture_screen(width=1280, height=720, region=None):
    """Capture screen using mss and save to disk."""
    try:
        print("Initializing mss screen capture...")
        with mss() as sct:
            # Print available monitors
            print(f"Available monitors: {sct.monitors}")
            
            # Setup monitor to capture
            if region:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
                print(f"Capturing region: {monitor}")
            else:
                # Capture the first monitor
                monitor = sct.monitors[1]  # 1 is the first monitor (0 is all monitors combined)
                print(f"Capturing full monitor: {monitor}")
            
            # Capture the screen
            print("Taking screenshot...")
            sct_img = sct.grab(monitor)
            print(f"Screenshot taken with size: {sct_img.size}, format: {sct_img.pixel_format}")
            
            # Convert to numpy array
            img = np.array(sct_img)
            print(f"Converted to numpy array with shape: {img.shape}, dtype: {img.dtype}")
            
            # Resize if specified dimensions differ from captured dimensions
            if (width != img.shape[1] or height != img.shape[0]) and width > 0 and height > 0:
                print(f"Resizing from {img.shape[1]}x{img.shape[0]} to {width}x{height}")
                img = cv2.resize(img, (width, height))
            
            # Create timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_capture_{timestamp}.png"
            
            # Save the image
            print(f"Saving screenshot to {filename}...")
            cv2.imwrite(filename, img)
            
            # Print success message with full path
            absolute_path = os.path.abspath(filename)
            print(f"\nScreenshot saved successfully to:\n{absolute_path}")
            return absolute_path
            
    except Exception as e:
        import traceback
        print(f"Error capturing screen: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Simple MSS Screen Capture")
    
    parser.add_argument('--width', type=int, default=1280, 
                      help='Width to resize the capture (use 0 for no resize)')
    parser.add_argument('--height', type=int, default=720, 
                      help='Height to resize the capture (use 0 for no resize)')
    parser.add_argument('--region', type=str, default=None, 
                      help='Region to capture: x,y,width,height')
    
    args = parser.parse_args()
    
    # Parse region if specified
    region = None
    if args.region:
        try:
            region = list(map(int, args.region.split(',')))
            if len(region) != 4:
                print("Region must be specified as: x,y,width,height")
                return
        except ValueError:
            print("Invalid region format. Use: x,y,width,height")
            return
    
    # Capture and save
    capture_screen(args.width, args.height, region)

if __name__ == "__main__":
    main()