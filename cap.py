from PIL import Image
import numpy as np
import time

# Global variable to store cached window handles
CACHED_HANDLES = {}
last_time_logged = 0

def capture_win_alt(convert: bool = False, window_name: str = "MegaMan_BattleNetwork_LegacyCollection_Vol2"):
    """
    Capture a screenshot of a window by name or partial match.
    
    Args:
        convert: Whether to convert the image (not used)
        window_name: The name of the window to capture
        
    Returns:
        numpy.ndarray: The captured image as a numpy array
    """
    global CACHED_HANDLES, last_time_logged
    
    # Import windows-specific modules
    from ctypes import windll
    import win32gui
    import win32ui

    # Check if we have a cached handle for this window
    cached_data = CACHED_HANDLES.get(window_name)
    
    # If no cached data, create new handles
    if cached_data is None:
        # Throttle logging to once per second
        current_time = time.time()
        if current_time - last_time_logged > 1.0:
            last_time_logged = current_time
            
        # Make app DPI aware to get correct sizes
        windll.user32.SetProcessDPIAware()
        
        # Try to find window by exact name
        hwnd = win32gui.FindWindow(None, window_name)
        
        # If exact match fails, try partial match
        if hwnd == 0:
            windows = []
            win32gui.EnumWindows(lambda h, extra: extra.append((h, win32gui.GetWindowText(h))), windows)
            
            for h, title in windows:
                if (window_name.lower() in title.lower() and 
                    win32gui.IsWindowVisible(h) and 
                    not win32gui.GetWindowTextLength(h) == 0):
                    hwnd = h
                    if current_time - last_time_logged > 1.0:
                        print(f"Found window by partial match: '{title}'")
                    break
                    
        # If still no window found, raise exception
        if hwnd == 0:
            # Get list of visible windows for debugging
            visible_windows = []
            win32gui.EnumWindows(
                lambda h, lst: lst.append(win32gui.GetWindowText(h)) 
                if win32gui.IsWindowVisible(h) and win32gui.GetWindowTextLength(h) > 0 
                else None, 
                visible_windows
            )
            print("Available windows:", ", ".join(f"'{w}'" for w in visible_windows[:10]))
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Get client area dimensions
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top
        
        # Create device contexts and bitmap
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        
        # Cache window data
        CACHED_HANDLES[window_name] = {
            'hwnd': hwnd,
            'hwnd_dc': hwnd_dc,
            'mfc_dc': mfc_dc,
            'save_dc': save_dc,
            'bitmap': bitmap,
            'width': w,
            'height': h
        }
    else:
        # Extract data from cache
        hwnd = cached_data['hwnd']
        hwnd_dc = cached_data['hwnd_dc']
        mfc_dc = cached_data['mfc_dc']
        save_dc = cached_data['save_dc']
        bitmap = cached_data['bitmap']
        w = cached_data['width']
        h = cached_data['height']
    
    # Check if window still exists
    if not win32gui.IsWindow(hwnd):
        # Clean up resources
        try:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass
            
        # Remove from cache and try again
        CACHED_HANDLES.pop(window_name, None)
        return capture_win_alt(convert, window_name)
    
    # Select bitmap into DC
    save_dc.SelectObject(bitmap)

    # Try PrintWindow with PW_CLIENTONLY flag (3)
    result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
    
    # If that fails, try without PW_CLIENTONLY (1)
    if result != 1:
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)

    try:
        # Get bitmap data
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)

        # Create PIL image from bitmap
        im = Image.frombuffer(
            "RGB", 
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), 
            bmpstr, "raw", "BGRX", 0, 1
        )
        
        # Convert to numpy array and return
        return np.array(im).copy()
        
    except Exception as e:
        current_time = time.time()
        if current_time - last_time_logged > 1.0:
            print(f"Error capturing window: {e}")
            last_time_logged = current_time
            
        # Clean up resources and remove from cache
        try:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass
            
        CACHED_HANDLES.pop(window_name, None)
        
        # Return black image on error
        return np.zeros((h, w, 3), dtype=np.uint8)

# Only run test code when this file is executed directly
if __name__ == "__main__":
    img = capture_win_alt(True, "Minecraft")
    Image.fromarray(img).save("output.jpeg", "jpeg")
    print(img)