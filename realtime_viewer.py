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

# Import the capture function
from cap import capture_win_alt

# Import Windows API modules for input forwarding
import ctypes
import win32gui
import win32con
import win32api

# Define Windows input constants and structures for input forwarding
# Mouse input
MOUSEEVENTF_MOVE = 0x0001  # mouse move
MOUSEEVENTF_LEFTDOWN = 0x0002  # left button down
MOUSEEVENTF_LEFTUP = 0x0004  # left button up
MOUSEEVENTF_RIGHTDOWN = 0x0008  # right button down
MOUSEEVENTF_RIGHTUP = 0x0010  # right button up
MOUSEEVENTF_MIDDLEDOWN = 0x0020  # middle button down
MOUSEEVENTF_MIDDLEUP = 0x0040  # middle button up
MOUSEEVENTF_WHEEL = 0x0800  # wheel button rolled
MOUSEEVENTF_ABSOLUTE = 0x8000  # absolute move

# Keyboard input
KEYEVENTF_EXTENDEDKEY = 0x0001  # extended key
KEYEVENTF_KEYUP = 0x0002  # key up

# Input types
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Keyboard message parameter bits
RESERVED_BIT_25 = 0x02000000  # Reserved bit 25, must be 1
PREVIOUS_KEY_STATE = 0x08000000  # Bit 27, previous key state
TRANSITION_STATE = 0x10000000  # Bit 28, transition state
SCANCODE_MASK = 0x00FF0000  # Bits 16-23, scan code

# Define input structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        # Skipping HARDWAREINPUT as we don't need it
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUTUNION)
    ]

class RealESRGANApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.upsampler = None
        self.running = False
        self.processed_frames = 0
        self.start_time = time.time()

        # Store window handle for input forwarding
        self.target_hwnd = None
        self.mouse_captured = False
        self.mouse_pos = (0, 0)  # Store last mouse position
        self.try_direct_input = args.try_direct_input # Use alternative input method if needed
        self.debug_input = args.debug_input  # Print debug info for input forwarding

        # Configure the main window
        self.root.title("RealESRGAN Window Filter")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Set up the UI
        self.setup_ui()

        # Get target window handle
        if self.args.overlay_mode:
            self.find_target_window()

        # Initialize model in background thread
        self.status_var.set("Initializing model...")
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def find_target_window(self):
        """Find the target window handle for input forwarding."""
        try:
            self.target_hwnd = win32gui.FindWindow(None, self.args.window_name)
            if self.target_hwnd == 0:
                print(f"Warning: Could not find window '{self.args.window_name}'")
                self.update_status(f"Warning: Could not find window '{self.args.window_name}'")

                # Try to enumerate through windows to suggest possible matches
                windows = []
                win32gui.EnumWindows(lambda hwnd, extra: extra.append((hwnd, win32gui.GetWindowText(hwnd))), windows)

                # Print possible game windows for debugging
                print("Possible windows:")
                for hwnd, title in windows:
                    if title and len(title) > 0 and win32gui.IsWindowVisible(hwnd):
                        print(f"  - '{title}' (handle: {hwnd})")
            else:
                print(f"Found target window: {self.args.window_name}, handle: {self.target_hwnd}")
                self.update_status(f"Found target window: {self.args.window_name}")

                # Get window rect to calculate scaling for mouse input
                left, top, right, bottom = win32gui.GetWindowRect(self.target_hwnd)
                self.target_window_rect = (left, top, right, bottom)
                self.target_window_size = (right - left, bottom - top)

                # Get client rect to know the usable game area
                client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.target_hwnd)
                self.target_client_rect = (client_left, client_top, client_right, client_bottom)
                self.target_client_size = (client_right - client_left, client_bottom - client_top)

                print(f"Window rect: {self.target_window_rect}, Client rect: {self.target_client_rect}")
                print(f"Window borders: horizontal={left}, vertical={top}")
        except Exception as e:
            print(f"Error finding target window: {e}")
            self.target_hwnd = None

    def setup_input_forwarding(self):
        """Set up input event forwarding to the target window."""
        if not self.target_hwnd:
            print("Cannot set up input forwarding - no target window handle")
            return False

        try:
            # Capture all keyboard events - bind to root for global keyboard capture
            self.root.bind("<KeyPress>", self.on_key_press)
            self.root.bind("<KeyRelease>", self.on_key_release)

            # Capture mouse events - bind to enhanced canvas
            self.canvas_enhanced.bind("<Motion>", self.on_mouse_move)
            self.canvas_enhanced.bind("<Button>", self.on_mouse_button)
            self.canvas_enhanced.bind("<ButtonRelease>", self.on_mouse_release)
            self.canvas_enhanced.bind("<MouseWheel>", self.on_mouse_wheel)

            # Track focus events to know when to capture the mouse
            self.root.bind("<FocusIn>", self.on_focus_in)
            self.root.bind("<FocusOut>", self.on_focus_out)

            # Always capture mouse when in overlay mode
            if self.args.overlay_mode:
                self.mouse_captured = True

            return True
        except Exception as e:
            print(f"Error setting up input forwarding: {e}")
            return False

    def on_key_press(self, event):
        """Handle key press events and forward to target window."""
        if not self.target_hwnd:
            return

        # Skip F7 and q since those are used to exit the app
        if event.keysym == 'F7' or event.keysym == 'q':
            return

        try:
            # Convert Tkinter key event to Windows virtual key code
            vk_code = self.get_virtual_key_code(event)
            if vk_code:
                # Send key down event
                self.send_key_input(vk_code, False)
                return 'break'  # Prevent further processing
        except Exception as e:
            print(f"Error forwarding key press: {e}")

    def on_key_release(self, event):
        """Handle key release events and forward to target window."""
        if not self.target_hwnd:
            return

        # Skip F7 and q
        if event.keysym == 'F7' or event.keysym == 'q':
            return

        try:
            # Convert Tkinter key event to Windows virtual key code
            vk_code = self.get_virtual_key_code(event)
            if vk_code:
                # Send key up event
                self.send_key_input(vk_code, True)
                return 'break'  # Prevent further processing
        except Exception as e:
            print(f"Error forwarding key release: {e}")

    def get_virtual_key_code(self, event):
        """Convert Tkinter key event to Windows virtual key code."""
        # This is a simplified mapping - we'd need a more complete mapping for all keys
        # For games, we mostly care about WASD, arrow keys, spacebar, etc.

        # Try to get virtual key code from the event directly if possible
        if hasattr(event, 'keycode'):
            return event.keycode

        # Handle common keys
        key_map = {
            'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45, 'f': 0x46,
            'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
            'm': 0x4D, 'n': 0x4E, 'o': 0x4F, 'p': 0x50, 'q': 0x51, 'r': 0x52,
            's': 0x53, 't': 0x54, 'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58,
            'y': 0x59, 'z': 0x5A, '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33,
            '4': 0x34, '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            'space': 0x20, 'Return': 0x0D, 'Tab': 0x09, 'BackSpace': 0x08,
            'Shift_L': 0x10, 'Shift_R': 0x10, 'Control_L': 0x11, 'Control_R': 0x11,
            'Alt_L': 0x12, 'Alt_R': 0x12, 'Caps_Lock': 0x14, 'Escape': 0x1B,
            'Left': 0x25, 'Up': 0x26, 'Right': 0x27, 'Down': 0x28,
            'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73, 'F5': 0x74,
            'F6': 0x75, 'F7': 0x76, 'F8': 0x77, 'F9': 0x78, 'F10': 0x79,
            'F11': 0x7A, 'F12': 0x7B
        }

        # Try to find the key in the map
        key = event.keysym.lower() if event.keysym else ''
        return key_map.get(event.keysym, 0)

    def send_key_input(self, vk_code, is_key_up):
        """Send a keyboard input event to the target window."""
        try:
            # Make sure the target window exists
            if not self.target_hwnd:
                return

            # Define the messages
            WM_KEYDOWN = 0x0100
            WM_KEYUP = 0x0101

            # Send key directly using SendMessage instead of PostMessage for more reliability
            msg = WM_KEYUP if is_key_up else WM_KEYDOWN

            # Get scan code from virtual key (more accurate for some games)
            try:
                scan_code = win32api.MapVirtualKey(vk_code, 0) << 16
            except Exception:
                scan_code = 0

            # Construct lParam correctly according to Windows API specifications
            # Bits 0-15: Repeat count (1 for single press)
            # Bits 16-23: Scan code
            # Bit 24: Extended key flag (0 for standard keys)
            # Bit 25: Reserved, must be 1
            # Bit 26: Context code (0 usually)
            # Bit 27: Previous key state (1 for key up events)
            # Bit 28: Transition state (1 for key up, 0 for key down)
            # Bits 29-31: Reserved, must be 0

            # Start with repeat count=1 and reserved bit 25 set to 1
            lParam = 1 | RESERVED_BIT_25 | scan_code

            # For key up events, also set previous key state and transition state
            if is_key_up:
                lParam |= PREVIOUS_KEY_STATE | TRANSITION_STATE

            # Try SendMessage for synchronous input
            try:
                # Send the key message directly - works better for games
                result = win32gui.SendMessage(self.target_hwnd, msg, vk_code, lParam)

                # If SendMessage failed, try PostMessage
                if result == 0:
                    win32gui.PostMessage(self.target_hwnd, msg, vk_code, lParam)
            except Exception:
                # Fallback to PostMessage
                win32gui.PostMessage(self.target_hwnd, msg, vk_code, lParam)

            # Try alternative direct input method for keyboard events
            # This sends to the active window, so we don't use it by default
            # but it might help with some games
            if self.args.forward_input and hasattr(self, 'try_direct_input') and self.try_direct_input:
                try:
                    # Create keyboard input structure
                    extra = ctypes.c_ulong(0)
                    ii_ = INPUT_KEYBOARD
                    flags = KEYEVENTF_KEYUP if is_key_up else 0

                    # Create input structure with scan code if available
                    scan = win32api.MapVirtualKey(vk_code, 0)
                    kb = KEYBDINPUT(vk_code, scan, flags, 0, ctypes.pointer(extra))
                    inp = INPUT(INPUT_KEYBOARD, INPUTUNION(ki=kb))

                    # Send the input
                    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
                except Exception:
                    pass

        except Exception as e:
            print(f"Error sending key input: {e}")

    def on_mouse_move(self, event):
        """Handle mouse movement events and forward to target window."""
        if not self.target_hwnd:
            return

        # Throttle mouse move events to avoid UI lag
        now = time.time()
        if hasattr(self, 'last_mouse_move') and now - self.last_mouse_move < 0.05:  # 50ms throttle
            return 'break'

        self.last_mouse_move = now

        try:
            # Get the current mouse position in the canvas
            x, y = event.x, event.y

            # Calculate the corresponding position in the target window
            target_x, target_y = self.map_coordinates_to_target(x, y)

            # Only send mouse move if significant position change (reduces jitter)
            if (not hasattr(self, 'mouse_pos') or
                abs(x - self.mouse_pos[0]) > 3 or
                abs(y - self.mouse_pos[1]) > 3):

                # Send mouse move event
                self.send_mouse_move(target_x, target_y)

                # Store the current position
                self.mouse_pos = (x, y)

            # Return 'break' to prevent default processing
            return 'break'

        except Exception as e:
            print(f"Error forwarding mouse move: {e}")

        return 'break'

    def on_mouse_button(self, event):
        """Handle mouse button press events and forward to target window."""
        if not self.target_hwnd:
            return

        # Capture the mouse on first click
        self.mouse_captured = True

        try:
            # Get the current mouse position in the canvas
            x, y = event.x, event.y

            # Calculate the corresponding position in the target window
            target_x, target_y = self.map_coordinates_to_target(x, y)

            # Determine which button was pressed
            if event.num == 1:  # Left button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_LEFTDOWN)
            elif event.num == 3:  # Right button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_RIGHTDOWN)
            elif event.num == 2:  # Middle button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_MIDDLEDOWN)

            return 'break'  # Prevent further processing

        except Exception as e:
            print(f"Error forwarding mouse button: {e}")

    def on_mouse_release(self, event):
        """Handle mouse button release events and forward to target window."""
        if not self.target_hwnd or not self.mouse_captured:
            return

        try:
            # Get the current mouse position in the canvas
            x, y = event.x, event.y

            # Calculate the corresponding position in the target window
            target_x, target_y = self.map_coordinates_to_target(x, y)

            # Determine which button was released
            if event.num == 1:  # Left button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_LEFTUP)
            elif event.num == 3:  # Right button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_RIGHTUP)
            elif event.num == 2:  # Middle button
                self.send_mouse_click(target_x, target_y, MOUSEEVENTF_MIDDLEUP)

            return 'break'  # Prevent further processing

        except Exception as e:
            print(f"Error forwarding mouse release: {e}")

    def on_mouse_wheel(self, event):
        """Handle mouse wheel events and forward to target window."""
        if not self.target_hwnd or not self.mouse_captured:
            return

        try:
            # Get the current mouse position in the canvas
            x, y = event.x, event.y

            # Calculate the corresponding position in the target window
            target_x, target_y = self.map_coordinates_to_target(x, y)

            # Get the wheel delta
            delta = event.delta

            # Send mouse wheel event
            self.send_mouse_wheel(target_x, target_y, delta)

            return 'break'  # Prevent further processing

        except Exception as e:
            print(f"Error forwarding mouse wheel: {e}")

    def on_focus_in(self, event):
        """Handle focus in events."""
        if self.args.overlay_mode and self.args.hide_cursor:
            # Hide the cursor in overlay mode
            self.root.config(cursor="none")

    def on_focus_out(self, event):
        """Handle focus out events."""
        # Release mouse capture when window loses focus
        self.mouse_captured = False
        if self.args.overlay_mode and self.args.hide_cursor:
            # Show cursor when focus is lost
            self.root.config(cursor="")

    def map_coordinates_to_target(self, x, y):
        """Map coordinates from the canvas to the target window's client area."""
        if not self.target_hwnd:
            return x, y

        try:
            # Get the size of the canvas and the target window
            canvas_width = self.canvas_enhanced.winfo_width()
            canvas_height = self.canvas_enhanced.winfo_height()

            # Calculate the relative position (0.0 to 1.0)
            rel_x = x / canvas_width
            rel_y = y / canvas_height

            # Map to the target window's client area
            target_x = int(rel_x * self.target_client_size[0])
            target_y = int(rel_y * self.target_client_size[1])

            return target_x, target_y
        except Exception as e:
            print(f"Error mapping coordinates: {e}")
            return x, y

    def send_mouse_move(self, x, y):
        """Send a mouse move event to the target window."""
        try:
            # Make sure target window exists
            if not self.target_hwnd:
                return

            # For games, it's often better to use SendMessage which waits for processing
            # rather than PostMessage which is asynchronous

            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(self.target_hwnd)

            # Get client area
            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_right - client_left
            client_height = client_bottom - client_top

            # Limit coordinates to client area
            x = max(0, min(x, client_width - 1))
            y = max(0, min(y, client_height - 1))

            # Use SendMessage instead of PostMessage for reliability
            WM_MOUSEMOVE = 0x0200
            lparam = win32api.MAKELONG(x, y)

            # Directly send to window, not posting
            if hasattr(win32gui, 'SendMessage'):
                win32gui.SendMessage(self.target_hwnd, WM_MOUSEMOVE, 0, lparam)
            else:
                # Use PostMessage as fallback
                win32gui.PostMessage(self.target_hwnd, WM_MOUSEMOVE, 0, lparam)

        except Exception as e:
            print(f"Error sending mouse move: {e}")

    def send_mouse_click(self, x, y, button_flag):
        """Send a mouse button event to the target window."""
        try:
            # Make sure target window exists
            if not self.target_hwnd:
                return

            # Get client area dimensions
            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_right - client_left
            client_height = client_bottom - client_top

            # Limit coordinates to client area
            x = max(0, min(x, client_width - 1))
            y = max(0, min(y, client_height - 1))

            # Determine appropriate message for button action
            if button_flag == MOUSEEVENTF_LEFTDOWN:
                msg = 0x0201  # WM_LBUTTONDOWN
            elif button_flag == MOUSEEVENTF_LEFTUP:
                msg = 0x0202  # WM_LBUTTONUP
            elif button_flag == MOUSEEVENTF_RIGHTDOWN:
                msg = 0x0204  # WM_RBUTTONDOWN
            elif button_flag == MOUSEEVENTF_RIGHTUP:
                msg = 0x0205  # WM_RBUTTONUP
            elif button_flag == MOUSEEVENTF_MIDDLEDOWN:
                msg = 0x0207  # WM_MBUTTONDOWN
            elif button_flag == MOUSEEVENTF_MIDDLEUP:
                msg = 0x0208  # WM_MBUTTONUP
            else:
                return

            # Set appropriate button flag
            wparam = 0x0001  # MK_LBUTTON
            if button_flag in (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP):
                wparam = 0x0002  # MK_RBUTTON
            elif button_flag in (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP):
                wparam = 0x0010  # MK_MBUTTON

            # Convert client coordinates to LPARAM
            lparam = win32api.MAKELONG(x, y)

            # Use SendMessage for synchronous processing
            if hasattr(win32gui, 'SendMessage'):
                win32gui.SendMessage(self.target_hwnd, msg, wparam, lparam)
            else:
                # Use PostMessage as fallback
                win32gui.PostMessage(self.target_hwnd, msg, wparam, lparam)

        except Exception as e:
            print(f"Error sending mouse click: {e}")

    def send_mouse_wheel(self, x, y, delta):
        """Send a mouse wheel event to the target window."""
        try:
            # Make sure target window exists
            if not self.target_hwnd:
                return

            # Convert the delta to the format expected by Windows
            wheel_delta = delta * 120  # Windows expects multiples of WHEEL_DELTA (120)

            # Get the current mouse position in client coordinates
            client_x, client_y = x, y

            # Convert to LPARAM format for PostMessage
            lparam = win32api.MAKELONG(client_x, client_y)

            # Try method 1: PostMessage - works for non-focused windows
            # WM_MOUSEWHEEL = 0x020A
            # The high word of wParam indicates the wheel delta
            # Bits 0-15 indicate which virtual keys are down (MK_CONTROL, MK_SHIFT, etc.)
            wparam = (wheel_delta << 16)  # Shift wheel delta to high word
            win32gui.PostMessage(self.target_hwnd, 0x020A, wparam, lparam)

            # Try method 2: SendInput - may work in some configurations
            extra = ctypes.c_ulong(0)
            ii_ = INPUT_MOUSE

            # Create the input structure with the wheel delta
            mi = MOUSEINPUT(0, 0, wheel_delta, MOUSEEVENTF_WHEEL, 0, ctypes.pointer(extra))
            inp = INPUT(ii_, INPUTUNION(mi=mi))

            # Send the input
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        except Exception as e:
            print(f"Error sending mouse wheel: {e}")

    def setup_ui(self):
        """Set up the user interface"""
        # Status bar
        self.status_var = tk.StringVar()

        if self.args.overlay_mode:
            # Overlay mode - fullscreen window with just the enhanced output
            # Configure fullscreen
            #self.root.attributes('-fullscreen', True)
            #self.root.overrideredirect(True)  # Removes window borders

            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_w}x{screen_h}+0+0")
            self.root.attributes('-topmost', True)  # Keep it above other windows

            # Create a frame to hold everything
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Create a canvas that fills the screen for the enhanced output
            self.canvas_enhanced = tk.Canvas(main_frame, bg='black')
            self.canvas_enhanced.pack(fill=tk.BOTH, expand=True)

            # Add a small status bar at the bottom
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

            # Add key bindings for exiting overlay mode - using F7 instead of Escape
            self.root.bind('<F7>', lambda e: self.on_close())
            self.root.bind('q', lambda e: self.on_close())

            # Hide cursor if requested
            if self.args.hide_cursor:
                self.root.config(cursor="none")

            # Setup input forwarding if enabled
            if self.args.forward_input and self.target_hwnd:
                self.setup_input_forwarding()

            # Start automatically in overlay mode
            self.root.after(100, self.start_capture)

            # Set variables needed for normal mode (even if we don't use them)
            self.canvas_orig = None
            self.downscale_var = tk.IntVar(value=self.args.downscale)
            self.outscale_var = tk.DoubleVar(value=self.args.outscale)
            self.fps_var = tk.StringVar(value="FPS: 0.0")
        else:
            # Normal mode with side-by-side comparison
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10 10 10 10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)

            # Status bar
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

            # Image display area - two canvases side by side
            self.display_frame = ttk.Frame(main_frame)
            self.display_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(0, weight=1)

            # Original image canvas
            self.canvas_orig = tk.Canvas(self.display_frame,
                                        width=self.args.width,
                                        height=self.args.height,
                                        bg='black')
            self.canvas_orig.grid(row=0, column=0, padx=5, pady=5)

            # Enhanced image canvas
            self.canvas_enhanced = tk.Canvas(self.display_frame,
                                            width=self.args.width,
                                            height=self.args.height,
                                            bg='black')
            self.canvas_enhanced.grid(row=0, column=1, padx=5, pady=5)

            # Canvas labels
            ttk.Label(self.display_frame, text="Original").grid(row=1, column=0)
            ttk.Label(self.display_frame, text="Enhanced").grid(row=1, column=1)

            # Control panel
            control_frame = ttk.Frame(main_frame)
            control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

            # Start button
            self.start_button = ttk.Button(control_frame, text="Start", command=self.start_capture, state=tk.DISABLED)
            self.start_button.grid(row=0, column=0, padx=5)

            # Stop button
            self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
            self.stop_button.grid(row=0, column=1, padx=5)

            # Placeholder for layout consistency
            ttk.Label(control_frame, text="").grid(row=0, column=2, padx=5)

            # FPS display
            self.fps_var = tk.StringVar(value="FPS: 0.0")
            ttk.Label(control_frame, textvariable=self.fps_var).grid(row=0, column=3, padx=20)

            # Downscale option
            self.downscale_var = tk.IntVar(value=self.args.downscale)
            ttk.Label(control_frame, text="Downscale:").grid(row=0, column=4, padx=(20,5))
            downscale_spin = ttk.Spinbox(control_frame, from_=1, to=4, width=5, textvariable=self.downscale_var)
            downscale_spin.grid(row=0, column=5)

            # Scale option
            self.outscale_var = tk.DoubleVar(value=self.args.outscale)
            ttk.Label(control_frame, text="Outscale:").grid(row=0, column=6, padx=(20,5))
            outscale_spin = ttk.Spinbox(control_frame, from_=0.5, to=4, increment=0.5, width=5, textvariable=self.outscale_var)
            outscale_spin.grid(row=0, column=7)

        # Current images
        self.current_original = None
        self.current_enhanced = None

    def initialize_model(self):
        """Initialize the RealESRGAN model in a background thread."""
        try:
            from basicsr.utils.download_util import load_file_from_url

            # Update status
            self.update_status("Setting up model...")

            # Set upsampler to None to prevent errors in process_frame during initialization
            self.upsampler = None

            # ---------------------- determine models according to model names ---------------------- #
            model_name = self.args.model_name.split('.pth')[0]

            # Create the model architecture
            if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif model_name.startswith('realesr-animevideov3'):  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            else:
                # Default model configuration
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
                netscale = 2

                # Use a reliable model path
                model_name = 'net_g_10000.pth'
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']

            # ---------------------- determine model paths ---------------------- #
            model_path = os.path.join('weights', model_name )

            print(f"model_path is {model_path}")


            # Model compilation happens in RealESRGANer class initialization
            # Check if torch.compile is available
            if hasattr(torch, 'compile'):
                self.update_status("Setting up model with torch.compile optimization...")
            else:
                self.update_status("torch.compile not available (requires PyTorch 2.0+)")

            print(f"Loading model from {model_path}")
            try:
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

                # Verify that upsampler was initialized correctly
                if self.upsampler is None:
                    raise RuntimeError("Failed to initialize upsampler")

                # Warmup with small image
                self.update_status("Warming up model...")
                dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                dummy_bgr = cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR)
                _, _ = self.upsampler.enhance(dummy_bgr, outscale=1.0)

                # Enable UI
                self.update_status("Ready")
                self.root.after(0, self.enable_ui)
            except Exception as e:
                print(f"Error initializing upsampler: {e}")
                traceback.print_exc()
                self.upsampler = None
                self.update_status(f"Error initializing upsampler: {str(e)}")
                raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"Error initializing model: {str(e)}")

    def update_status(self, message):
        """Update status bar message."""
        if self.args.overlay_mode and self.running:
            # In overlay mode and running, preserve FPS info
            elapsed = time.time() - self.start_time

            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            print(fps)
            self.root.after(0, lambda: self.status_var.set(f"{message} | FPS: {fps:.1f} | Press F7 or 'q' to exit"))

        else:
            self.root.after(0, lambda: self.status_var.set(message))

    def enable_ui(self):
        """Enable UI elements after initialization."""
        if not self.args.overlay_mode:
            self.start_button.config(state=tk.NORMAL)

    def capture_window(self):
        """Capture the target window using capture_win_alt."""
        try:
            # Capture the window specified by window_name
            window_name = self.args.window_name

            # If we found a window handle by partial match, use that window's title
            if self.target_hwnd:
                try:
                    actual_window_name = win32gui.GetWindowText(self.target_hwnd)
                    if actual_window_name:
                        window_name = actual_window_name
                except Exception:
                    pass

            # Only print this message once per second to reduce spam
            now = time.time()
            if not hasattr(self, 'last_capture_log') or now - self.last_capture_log > 1.0:
                self.last_capture_log = now

            # Capture window content
            img = capture_win_alt(convert=True, window_name=window_name)

            # Resize if needed
            if img.shape[1] != self.args.width or img.shape[0] != self.args.height:
                img = cv2.resize(img, (self.args.width, self.args.height))

            return img
        except Exception as e:
            print(f"Error capturing window: {e}")
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

    def process_frame(self, img, downscale, outscale):
        """Process a frame using RealESRGAN."""
        try:
            # Check if the upsampler has been properly initialized
            if self.upsampler is None:
                print("Warning: Upsampler not initialized yet, returning original image")
                if img.dtype == np.uint8 and img.shape[2] == 3:
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Return original in BGR
                return img

            # RGB to BGR conversion if needed
            # The capture function returns RGB array from PIL, so we need to convert
            if img.dtype == np.uint8 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img  # Assume already in the correct format

            # Apply downscaling if specified (to improve performance)
            if downscale > 1:
                proc_h, proc_w = img_bgr.shape[0] // downscale, img_bgr.shape[1] // downscale
                img_small = cv2.resize(img_bgr, (proc_w, proc_h))

                # Adjust outscale to compensate for the downscaling
                effective_outscale = outscale * downscale

                # Process using enhanced image with effective outscale
                output, _ = self.upsampler.enhance(img_small, outscale=effective_outscale)

            else:
                print(img_bgr.shape)

                #print(outscale)
                outscale = 2

                # Process the image at full resolution
                output, _ = self.upsampler.enhance(img_bgr, outscale=outscale)

                print(output.shape)

            if output is None:
                print("Warning: Enhancement produced None output, returning original image")
                return img_bgr

            return output
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            if img.dtype == np.uint8 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Return original in BGR
            return img

    def update_display(self, original_img, enhanced_img):
        """Update the display with original and enhanced images."""
        try:
            # Store current images
            self.current_original = original_img
            self.current_enhanced = enhanced_img

            # Convert enhanced image from BGR to RGB
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

            # Create PIL image for enhanced view
            pil_enhanced = Image.fromarray(enhanced_rgb)

            if self.args.overlay_mode:
                # In overlay mode, resize the enhanced image to fill the screen
                screen_width = self.root.winfo_width()
                screen_height = self.root.winfo_height()

                if screen_width <= 1 or screen_height <= 1:
                    # Window hasn't been drawn yet, use the specified dimensions
                    screen_width = 1920  # Default to standard HD width
                    screen_height = 1080  # Default to standard HD height

                # Resize while maintaining aspect ratio and filling the screen
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
                self.canvas_enhanced.delete("all")
                self.canvas_enhanced.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_enhanced)

                # Update the status bar with performance info
                elapsed = time.time() - self.start_time
                fps = self.processed_frames / elapsed if elapsed > 0 else 0
                self.status_var.set(f"Capturing {self.args.window_name} | FPS: {fps:.1f} | Press ESC or 'q' to exit")
            else:
                # Normal mode with side-by-side comparison
                # Create PIL image for original view
                pil_original = Image.fromarray(original_img)

                # Convert to Tkinter PhotoImage
                self.tk_original = ImageTk.PhotoImage(image=pil_original)
                self.tk_enhanced = ImageTk.PhotoImage(image=pil_enhanced)

                # Update canvases
                self.canvas_orig.create_image(0, 0, anchor=tk.NW, image=self.tk_original)
                self.canvas_enhanced.create_image(0, 0, anchor=tk.NW, image=self.tk_enhanced)
        except Exception as e:
            import traceback
            print(f"Error updating display: {e}")
            traceback.print_exc()

    def capture_loop(self):
        """Main capture and processing loop."""
        if not self.running:
            return

        try:
            frame_start_time = time.time()

            # Capture window
            original_img = self.capture_window()

            # Get current settings from UI
            downscale = self.downscale_var.get()
            outscale = self.outscale_var.get()

            # Process the image
            enhanced_img = self.process_frame(original_img, downscale, outscale)

            # Update display
            self.update_display(original_img, enhanced_img)

            # Update statistics
            self.processed_frames += 1
            frame_time = time.time() - frame_start_time
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0

            # Update FPS display every 5 frames
            if self.processed_frames % 5 == 0 and not self.args.overlay_mode:
                self.fps_var.set(f"FPS: {fps:.1f}")

            # Calculate delay to maintain target FPS
            if self.args.fps_target > 0:
                target_frame_time = 1.0 / self.args.fps_target
                delay = int(max(1, (target_frame_time - frame_time) * 1000))
            else:
                delay = 1  # Default to 1ms delay

            # Schedule next frame
            self.root.after(delay, self.capture_loop)

        except Exception as e:
            import traceback
            print(f"Error in capture loop: {e}")
            traceback.print_exc()
            # Try to recover
            self.root.after(1000, self.capture_loop)

    def start_capture(self):
        """Start the capture process."""
        if not self.running:
            # Always try to find/refresh the target window handle
            # regardless of overlay mode to enable input forwarding
            if not self.target_hwnd:
                self.find_target_window()

            # Make another attempt with partial window name matching if exact match failed
            if not self.target_hwnd:
                self.find_target_window_by_partial_match()

            # If we found a window handle, set up input forwarding
            if self.target_hwnd:
                self.setup_input_forwarding()
                # Do NOT make the game window the foreground window
                # This would hide our viewer window and defeat the purpose

                # Get the actual window title (in case we used partial matching)
                try:
                    actual_window_name = win32gui.GetWindowText(self.target_hwnd)
                    if actual_window_name:
                        print(f"Using window: '{actual_window_name}' (handle: {self.target_hwnd})")
                        self.args.window_name = actual_window_name
                except Exception:
                    pass

            self.running = True
            self.processed_frames = 0
            self.start_time = time.time()
            if not self.args.overlay_mode:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)

            status_msg = f"Capturing {self.args.window_name}"
            if self.args.overlay_mode:
                status_msg += " | Press F7 or 'q' to exit"

            if not self.target_hwnd:
                status_msg += " | WARNING: Target window not found! Input forwarding disabled."
            else:
                status_msg += f" | Input forwarding active"

            self.update_status(status_msg)
            self.capture_loop()

    def find_target_window_by_partial_match(self):
        """Find the target window handle by partial window title match."""
        try:
            # Store all windows with their titles
            windows = []
            win32gui.EnumWindows(lambda hwnd, extra: extra.append((hwnd, win32gui.GetWindowText(hwnd))), windows)

            # Filter for visible windows that partially match the target name
            search_term = self.args.window_name.lower()
            matches = []

            for hwnd, title in windows:
                if (title and len(title) > 0 and
                    win32gui.IsWindowVisible(hwnd) and
                    search_term in title.lower()):
                    matches.append((hwnd, title))

            if matches:
                # Sort by closest match length (prefer shorter matches that contain the search term)
                matches.sort(key=lambda x: len(x[1]))
                best_match_hwnd, best_match_title = matches[0]

                print(f"Found partial window match: '{best_match_title}' (handle: {best_match_hwnd})")
                self.target_hwnd = best_match_hwnd

                # Get window rect to calculate scaling for mouse input
                left, top, right, bottom = win32gui.GetWindowRect(self.target_hwnd)
                self.target_window_rect = (left, top, right, bottom)
                self.target_window_size = (right - left, bottom - top)

                # Get client rect to know the usable game area
                client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.target_hwnd)
                self.target_client_rect = (client_left, client_top, client_right, client_bottom)
                self.target_client_size = (client_right - client_left, client_bottom - client_top)

                print(f"Window rect: {self.target_window_rect}, Client rect: {self.target_client_rect}")
                return True

            else:
                print(f"No partial matches found for '{self.args.window_name}'")
                return False

        except Exception as e:
            print(f"Error finding target window by partial match: {e}")
            return False

    def stop_capture(self):
        """Stop the capture process."""
        self.running = False
        if not self.args.overlay_mode:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        self.update_status("Stopped")

    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN Window Enhancer with Live Preview")

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

    # Capture parameters
    parser.add_argument('--width', type=int, default=640,
                        help='Width of the display window')
    parser.add_argument('--height', type=int, default=480,
                        help='Height of the display window')
    parser.add_argument('--window_name', type=str, default="Minecraft",
                        help='Title of the window to capture')
    parser.add_argument('--downscale', type=int, default=1,
                       help='Downscale factor before processing (1=no downscale, 2=half resolution)')
    parser.add_argument('--fps_target', type=int, default=0,
                       help='Target FPS (0 = unlimited)')
    parser.add_argument('--dummy_frame', type=str, default=None,
                       help='Path to a JPEG image to use instead of window capture')

    # Display mode
    parser.add_argument('--overlay_mode', action='store_true',
                       help='Run in overlay mode (fullscreen with enhanced output only)')

    # Input forwarding
    parser.add_argument('--forward_input', action='store_true',
                       help='Forward keyboard and mouse input to the target window')
    parser.add_argument('--hide_cursor', action='store_true',
                       help='Hide the cursor in overlay mode')
    parser.add_argument('--try_direct_input', action='store_true',
                       help='Try alternative direct input method (may help with some games)')
    parser.add_argument('--debug_input', action='store_true',
                       help='Print debug information for input forwarding')

    args = parser.parse_args()

    # In overlay mode, automatically enable input forwarding unless explicitly disabled
    if args.overlay_mode and not parser.get_default('forward_input'):
        args.forward_input = True

    # Create and run the application
    root = tk.Tk()

    app = RealESRGANApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()