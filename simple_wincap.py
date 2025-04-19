from windows_capture import WindowsCapture, Frame, InternalCaptureControl

# Create the simplest possible capture
capture = WindowsCapture(
    cursor_capture=True,  # Include cursor in the captured frames
    draw_border=False,    # Don't draw a border around the captured area
    monitor_index=1,      # Use the primary monitor
    window_name=None      # Capture entire monitor
)

# Event handlers
@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    print("New frame arrived")
    
    # Save the frame as an image
    frame.save_as_image("screenshot.png")
    print("Saved screenshot to screenshot.png")
    
    # Stop after capturing one frame
    capture_control.stop()

@capture.event
def on_closed():
    print("Capture session closed")

if __name__ == "__main__":
    print("Starting capture...")
    capture.start()
    print("Capture complete")