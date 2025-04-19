import argparse
import cv2
import numpy as np
import time
import datetime
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

def save_frame(img, prefix="capture"):
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
    parser = argparse.ArgumentParser(description="Windows Capture Test")
    parser.add_argument("--monitor", type=int, default=1,
                      help="Monitor index to capture (1-based)")
    parser.add_argument("--frames", type=int, default=1,
                      help="Number of frames to capture before exiting")
    parser.add_argument('--region', type=str, default=None,
                      help='Region to crop after capture: x,y,width,height')

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

    # Frame counter
    frame_count = 0
    running = True

    print(f"Initializing capture for monitor {args.monitor}...")

    # Create capture instance
    capture = WindowsCapture(
        cursor_capture=True,
        draw_border=False,
        monitor_index=args.monitor,
        window_name=None
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        nonlocal frame_count, running

        if not running:
            capture_control.stop()
            return

        try:
            # Get frame data from frame_buffer
            img = frame.frame_buffer
            print(f"Captured frame {frame_count+1} with shape: {img.shape}, dtype: {img.dtype}")

            # Crop if region specified
            if region:
                x, y, w, h = region
                if x + w <= img.shape[1] and y + h <= img.shape[0]:
                    img = img[y:y+h, x:x+w]
                    print(f"Cropped to region: {region}, new shape: {img.shape}")
                else:
                    print(f"Region {region} is outside of capture bounds {img.shape[1]}x{img.shape[0]}")

            # Save frame
            save_frame(img, f"wincap_test_{frame_count+1}")

            # Increment counter
            frame_count += 1

            # Stop if reached frame limit
            if frame_count >= args.frames:
                print(f"Captured {args.frames} frames, stopping")
                running = False
                capture_control.stop()

        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()

    @capture.event
    def on_closed():
        print("Capture session closed")

    # Start capture
    print("Starting capture...")
    capture.start()

    # Wait for completion
    print("Waiting for capture to complete...")
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        running = False

if __name__ == "__main__":
    main()