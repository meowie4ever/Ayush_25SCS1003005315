"""
Hand Gesture Control System
===========================

A webcam-based hand gesture control project using:
- OpenCV for video capture and drawing
- MediaPipe for hand landmark detection
- PyAutoGUI for mouse and scrolling control

Gestures implemented:
- Index finger only -> move mouse
- Fist -> left click
- Pinch (thumb + index close) -> right click
- Index + middle fingers -> scroll

Controls:
- Press 'q' to quit
- Press 'p' to pause/resume
- Press 'r' to reset the gesture state

Install:
    pip install opencv-python mediapipe pyautogui numpy

Notes:
- pyautogui works best on a desktop environment with an active display.
- On Linux, you may need additional system packages for GUI access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# Safer defaults for mouse automation
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


@dataclass
class GestureState:
    gesture: str = "none"
    last_action_time: float = 0.0
    paused: bool = False
    prev_scroll_y: float | None = None


class HandGestureController:
    def __init__(
        self,
        camera_index: int = 0,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
        smoothening: float = 0.20,
        click_cooldown: float = 0.8,
        scroll_sensitivity: int = 40,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothening = smoothening
        self.click_cooldown = click_cooldown
        self.scroll_sensitivity = scroll_sensitivity
        self.state = GestureState()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.prev_x = 0
        self.prev_y = 0
        self.curr_x = 0
        self.curr_y = 0

    @staticmethod
    def _landmarks_to_pixels(hand_landmarks, frame_w: int, frame_h: int) -> List[Tuple[int, int]]:
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        return pts

    @staticmethod
    def _finger_states(lm_px: List[Tuple[int, int]], handedness_label: str) -> Dict[str, bool]:
        # Finger tip indices in MediaPipe Hands
        tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        pip = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

        states = {}
        # Thumb direction depends on hand label
        if handedness_label.lower() == "right":
            states["thumb"] = lm_px[tips["thumb"]][0] > lm_px[pip["thumb"]][0]
        else:
            states["thumb"] = lm_px[tips["thumb"]][0] < lm_px[pip["thumb"]][0]

        for finger in ["index", "middle", "ring", "pinky"]:
            states[finger] = lm_px[tips[finger]][1] < lm_px[pip[finger]][1]

        return states

    @staticmethod
    def _pinch_distance(lm_px: List[Tuple[int, int]]) -> float:
        x1, y1 = lm_px[4]
        x2, y2 = lm_px[8]
        return float(np.hypot(x2 - x1, y2 - y1))

    def _classify_gesture(self, states: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> str:
        thumb, index, middle, ring, pinky = (
            states["thumb"],
            states["index"],
            states["middle"],
            states["ring"],
            states["pinky"],
        )

        pinch = self._pinch_distance(lm_px) < 35  # pixels; works reasonably for a typical webcam setup

        if pinch and index:
            return "pinch"
        if not any([thumb, index, middle, ring, pinky]):
            return "fist"
        if index and middle and not ring and not pinky:
            return "two_fingers"
        if index and not middle and not ring and not pinky:
            return "point"
        if thumb and index and middle and ring and pinky:
            return "open_palm"
        return "unknown"

    def _move_mouse(self, x: int, y: int) -> None:
        # Smooth movement to reduce jitter
        self.curr_x = self.prev_x + (x - self.prev_x) * self.smoothening
        self.curr_y = self.prev_y + (y - self.prev_y) * self.smoothening

        pyautogui.moveTo(self.screen_w - self.curr_x, self.curr_y, duration=0)
        self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def _handle_action(self, gesture: str, lm_px: List[Tuple[int, int]], frame_w: int, frame_h: int) -> None:
        now = time.time()

        if gesture == "point":
            # Use index fingertip to move the cursor
            ix, iy = lm_px[8]
            x = np.interp(ix, (0, frame_w), (0, self.screen_w))
            y = np.interp(iy, (0, frame_h), (0, self.screen_h))
            self._move_mouse(int(x), int(y))

        elif gesture == "fist":
            if now - self.state.last_action_time >= self.click_cooldown:
                pyautogui.click()
                self.state.last_action_time = now

        elif gesture == "pinch":
            if now - self.state.last_action_time >= self.click_cooldown:
                pyautogui.rightClick()
                self.state.last_action_time = now

        elif gesture == "two_fingers":
            # Scroll based on vertical movement of the midpoint between index and middle fingertips
            ix, iy = lm_px[8]
            mx, my = lm_px[12]
            mid_y = (iy + my) / 2.0

            if self.state.prev_scroll_y is not None:
                delta = self.state.prev_scroll_y - mid_y
                steps = int(delta / max(1, self.scroll_sensitivity))
                if steps != 0:
                    pyautogui.scroll(steps * 120)
            self.state.prev_scroll_y = mid_y

        else:
            self.state.prev_scroll_y = None

    def run(self) -> None:
        print("Hand Gesture Controller started.")
        print("Press 'q' to quit, 'p' to pause/resume, 'r' to reset state.")

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            status_text = "No hand detected"

            if self.state.paused:
                cv2.putText(frame, "PAUSED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness_label = results.multi_handedness[0].classification[0].label
                    lm_px = self._landmarks_to_pixels(hand_landmarks, frame_w, frame_h)
                    states = self._finger_states(lm_px, handedness_label)
                    gesture = self._classify_gesture(states, lm_px)
                    self.state.gesture = gesture

                    self._handle_action(gesture, lm_px, frame_w, frame_h)

                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

                    status_text = f"{handedness_label} hand | Gesture: {gesture}"
                    for idx, (x, y) in enumerate(lm_px):
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), cv2.FILLED)
                        if idx in [4, 8, 12, 16, 20]:
                            cv2.putText(
                                frame,
                                str(idx),
                                (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),
                                1,
                            )
                else:
                    self.state.prev_scroll_y = None

            cv2.rectangle(frame, (10, 10), (520, 100), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, "Hand Gesture Control System", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                self.state.paused = not self.state.paused
            elif key == ord("r"):
                self.state = GestureState()

        self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    try:
        controller = HandGestureController()
        controller.run()
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
