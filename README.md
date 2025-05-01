
# AI-Powered Minecraft Shader

This project implements a real-time lighting enhancement system for Minecraft using a fine-tuned RealESRGAN model. It aims to improve the visual quality of the game by applying AI‑based style transfer to enhance lighting effects.

For more details, see the [technical paper](#).

---

## Demo

Watch a demonstration of the AI shader in action:
[Google Drive Video Link]([https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://drive.google.com/file/d/1xlMLYF8YKRFOQ0o5rnbmJutuw2WTQNnR/view?usp=sharing))

---

## Installation

```bash
# Clone the repository
git clone https://github.com/USERNAME/minecraft-ai-shader.git

# Navigate to the project directory
cd minecraft-ai-shader

# Install the required dependencies
pip install -r requirements.txt
```

---

## Usage

1. Ensure Minecraft is running in windowed mode at **1280×720** resolution on **Windows 11**.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The script will capture frames from the Minecraft window, apply the AI shader, and display the enhanced frames in a separate window.

---

## Features

- **Real-time enhancement** of Minecraft graphics using AI.
- Utilizes a fine‑tuned RealESRGAN model for style transfer.
- Runs on consumer‑grade hardware (e.g., RTX 4070 mobile GPUs).

---

## Requirements

- **Operating System:** Windows 11
- **Python:** 3.10
- **Frameworks & Libraries:**
  - PyTorch 2.0
  - TorchVision
  - Real‑ESRGAN (SRVGGNet x4) model
  - pywin32 and pyautogui (for window capture and input forwarding)
- **GPU Acceleration:** CUDA 12.1 and cuDNN
- **Minecraft:** Installed and running in windowed mode at 1280×720

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes, including tests if applicable
4. Submit a Pull Request and ensure your code follows the project’s coding standards

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The RealESRGAN team for their upscaling model
- NVIDIA for their advancements in AI and graphics
- The Minecraft community for inspiration and support
