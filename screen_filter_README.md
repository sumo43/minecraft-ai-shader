# RealESRGAN Screen Filter

This tool creates a real-time screen filter using the RealESRGAN model, allowing you to enhance content displayed on your screen using AI upscaling technology.

## Requirements

- Python 3.8+
- PyTorch with CUDA (for best performance)
- The required packages:
  ```
  pip install pyautogui opencv-python pillow torch basicsr
  ```
- Download the appropriate model weights from the Real-ESRGAN releases and place them in the `weights` folder

## Usage

1. Make sure you have the model weights downloaded in the `weights` folder
2. Run the screen filter with:
   ```
   python screen_filter.py
   ```

## Command Line Arguments

- `--model_name`: Model to use (default: 'realesr-animevideov3')
- `--outscale`: Output scale factor (default: 1.0)
- `--width`: Width of capture window (default: 640)
- `--height`: Height of capture window (default: 480)
- `--region`: Specific region to capture (format: "x,y,width,height")
- `--fps`: Target FPS (default: 30)
- `--tile`: Tile size for processing large images (default: 0)
- `--fp32`: Use FP32 precision instead of FP16 half precision

## Examples

Capture full screen at 720p:
```
python screen_filter.py --width 1280 --height 720
```

Capture a specific region (e.g., a video player):
```
python screen_filter.py --region "100,100,800,600"
```

Use a different model:
```
python screen_filter.py --model_name RealESRGAN_x4plus
```

## Controls

- **Start**: Begin capturing and processing the screen
- **Stop**: Pause the screen capture

## Performance Tips

1. Lower the capture resolution for better FPS
2. Use a smaller region instead of full screen
3. If you have a CUDA-capable GPU, make sure PyTorch is using it
4. For slower computers, try using `--outscale 0.5` to get smoother performance

## Extending this Tool

This is a basic implementation that can be extended in several ways:
- Add overlay capability to place the enhanced output back on the screen
- Implement more sophisticated region selection
- Add recording capability
- Add more advanced filter controls