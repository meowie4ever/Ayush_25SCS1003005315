import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import urllib.request
import os

# 1. Automatically download the required AI model if it's missing
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading Google's Hand Tracking Model... (This only happens once)")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the new MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Get screen resolution and configure mouse
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False 

# --- Gesture Logic ---
def is_pointing(landmarks):
    """Returns True if only the index finger is extended."""
    index_up = landmarks[8].y < landmarks[6].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    return index_up and middle_down and ring_down and pinky_down

def is_fist(landmarks):
    """Returns True if all fingers are curled into a fist."""
    index_down = landmarks[8].y > landmarks[6].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    return index_down and middle_down and ring_down and pinky_down

def is_open_hand(landmarks):
    """Returns True if all fingers are extended (used for scrolling)."""
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return index_up and middle_up and ring_up and pinky_up

def main():
    cap = cv2.VideoCapture(0)
    prev_y = None
    
    print("Starting Camera... Press 'q' in the video window to quit.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect hands
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Get the first detected hand
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # Draw simple dots on the joints so you can see the tracking
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                
                current_y = hand_landmarks[0].y
                
                # 1. Pointing -> Move Mouse
                if is_pointing(hand_landmarks):
                    index_tip = hand_landmarks[8]
                    x = int(index_tip.x * screen_width)
                    y = int(index_tip.y * screen_height)
                    pyautogui.moveTo(x, y, _pause=False)
                    cv2.putText(frame, "Moving Mouse", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    prev_y = None 
                    
                # 2. Fist -> Click Mouse
                elif is_fist(hand_landmarks):
                    pyautogui.click(_pause=False)
                    cv2.putText(frame, "Clicking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    prev_y = None 
                    cv2.waitKey(250) 
                    
                # 3. Open Hand -> Scroll
                elif is_open_hand(hand_landmarks):
                    cv2.putText(frame, "Scrolling Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    if prev_y is not None:
                        y_diff = current_y - prev_y
                        if abs(y_diff) > 0.02: 
                            scroll_speed = int((prev_y - current_y) * 3000)
                            pyautogui.scroll(scroll_speed)
                    prev_y = current_y 
                else:
                    prev_y = None
            
            cv2.imshow('Hand Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
