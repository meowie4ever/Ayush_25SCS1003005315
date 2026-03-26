# 🖐️ Hand Gesture Control

A Python application that uses computer vision to detect hand gestures and control your mouse cursor, perform clicks, and scroll using just your hand.

## ✨ Features

- 🖱️ **Mouse Movement**: Point your index finger to move the mouse cursor
- 👆 **Mouse Clicking**: Make a fist to click
- 📜 **Scrolling**: Open your hand to scroll up/down
- �� **Real-time Tracking**: Uses Google's MediaPipe Hand Landmarker for accurate hand detection
- ⚡ **Auto Model Download**: Automatically downloads the required AI model on first run

## 📋 Requirements

- 🐍 Python 3.7+
- 📷 Webcam
- 💻 Windows, macOS, or Linux

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe pyautogui
```

### 2. Run the Application

```bash
python main.py
```

The application will automatically download Google's Hand Landmarker model (~340 MB) on first run.

## 🎮 Usage

1. **Start the program**: `python main.py`
2. **Point your index finger** 👉 to move the mouse cursor
3. **Make a fist** ✊ to click the mouse button
4. **Open your hand** ✋ (all fingers extended) to scroll
5. **Press 'q'** ⌨️ in the video window to quit

## 👆 Gesture Recognition

### 👉 Pointing Gesture
- **Trigger**: Index finger extended, all other fingers curled
- **Action**: Moves mouse cursor to your index finger position
- **Use Case**: Navigate menus, hover over buttons

### ✊ Fist Gesture
- **Trigger**: All fingers curled down
- **Action**: Performs a mouse click
- **Use Case**: Click buttons, select items

### ✋ Open Hand Gesture
- **Trigger**: All fingers extended upward
- **Action**: Scrolls the page (hand position determines scroll direction and speed)
- **Use Case**: Scroll up/down through documents or web pages

## ⚙️ How It Works

1. **Video Capture**: 📹 Captures video from your webcam in real-time
2. **Hand Detection**: 🤖 Uses MediaPipe Hand Landmarker to detect hand landmarks
3. **Gesture Classification**: 🧠 Analyzes finger positions to determine the current gesture
4. **Action Execution**: 
   - Maps hand position to screen coordinates for mouse movement
   - Sends click commands when fist is detected
   - Calculates scroll distance based on hand movement

## 🔧 Troubleshooting

### ❌ Poor Hand Detection
- Ensure adequate lighting in your environment
- Keep your hand clearly visible to the camera
- Avoid strong backlighting

### 🐭 Shaky Mouse Movement
- Keep your hand steady
- Reduce background clutter
- Try adjusting the sensitivity by modifying the `0.02` threshold in the open hand scrolling logic

### 🎯 Mouse Pointer Jumping
- Ensure your webcam is positioned at a comfortable angle
- Check that you're within the camera's field of view
- Try reducing ambient motion in the background

### 📥 Model Download Issues
- Ensure you have a stable internet connection
- The model file (~340 MB) will be saved as `hand_landmarker.task` in the project directory
- If download fails, manually download from: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

## ⚡ Customization

### Adjust Scroll Sensitivity
Modify line 107 in `main.py`:
```python
scroll_speed = int((prev_y - current_y) * 3000)  # Change 3000 to adjust sensitivity
```

### Adjust Scroll Threshold
Modify line 105 in `main.py`:
```python
if abs(y_diff) > 0.02:  # Change 0.02 to adjust threshold
```

### Detect Multiple Hands
Modify line 14 in `main.py`:
```python
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)  # Change to detect more hands
```

## 🔬 Technical Details

- **Framework**: 🏗️ MediaPipe Tasks (Google's machine learning framework)
- **Computer Vision**: 👁️ Hand pose estimation using 21-point landmark detection
- **Mouse Control**: 🎮 PyAutoGUI for cross-platform mouse and keyboard automation
- **Video Processing**: 🎬 OpenCV for real-time video capture and processing

## ⚠️ Limitations

- Works best with one hand at a time (currently configured for single-hand detection)
- Requires good lighting and clear hand visibility
- May not work accurately with very small or large hands relative to camera distance
- Sensitivity to background motion

## 📄 License

This project uses open-source libraries. See individual library licenses for details.

## 🚀 Future Improvements

- [ ] 🙌 Multi-hand support for two-handed gestures
- [ ] 🤟 Additional gesture types (pinch, thumbs up, etc.)
- [ ] ⚙️ Configurable gesture mapping
- [ ] ⚡ Performance optimization for lower-end systems
- [ ] 🎓 Gesture recording and custom gesture training
