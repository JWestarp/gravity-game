"""
Hand Tracker Module for Gravity Game
Hand detection using MediaPipe
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os


class HandTracker:
    # Hand Landmark Indizes
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(self):
        # Download model file if not present
        model_path = self._ensure_model()
        
        # Configure MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        
        # Smoothing for fluid movements
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.smoothing_factor = 0.3  # Lower value = more responsive
        
        # Last frame for overlay
        self.last_frame = None
        self.last_landmarks = None
    
    def _ensure_model(self):
        """Ensures the hand landmark model is available"""
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded.")
        
        return model_path
        
    def start_camera(self, camera_index=0):
        """Starts the webcam"""
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False
    
    def stop_camera(self):
        """Stops the webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_hand_position(self):
        """
        Reads a frame and returns the hand position.
        
        Returns:
            tuple: (x, y, gesture) or None if no hand detected
                   x, y: Normalized position (0-1)
                   gesture: 'point', 'fist' or 'unknown'
        """
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip for more natural control
        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand
        results = self.detector.detect(mp_image)
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            landmarks = results.hand_landmarks[0]
            self.last_landmarks = landmarks
            
            # Use index finger tip position for precise control
            index_tip = landmarks[self.INDEX_FINGER_TIP]
            
            raw_x = index_tip.x
            raw_y = index_tip.y
            
            # Apply smoothing (exponential moving average)
            self.smooth_x = self.smooth_x * self.smoothing_factor + raw_x * (1 - self.smoothing_factor)
            self.smooth_y = self.smooth_y * self.smoothing_factor + raw_y * (1 - self.smoothing_factor)
            
            # Detect gesture
            gesture = self._detect_gesture(landmarks)
            
            return (self.smooth_x, self.smooth_y, gesture)
        
        self.last_landmarks = None
        return None
    
    def get_hand_overlay(self, target_width, target_height, alpha=0.3):
        """
        Returns hand landmarks as line skeleton.
        
        Args:
            target_width: Target overlay width
            target_height: Target overlay height
            alpha: Transparency (0-1)
        
        Returns:
            List of lines [(start, end), ...] and points, or None
        """
        if self.last_landmarks is None:
            return None
        
        # Define hand connections
        connections = [
            # Thumb
            (self.WRIST, self.THUMB_CMC),
            (self.THUMB_CMC, self.THUMB_MCP),
            (self.THUMB_MCP, self.THUMB_IP),
            (self.THUMB_IP, self.THUMB_TIP),
            # Index finger
            (self.WRIST, self.INDEX_FINGER_MCP),
            (self.INDEX_FINGER_MCP, self.INDEX_FINGER_PIP),
            (self.INDEX_FINGER_PIP, self.INDEX_FINGER_DIP),
            (self.INDEX_FINGER_DIP, self.INDEX_FINGER_TIP),
            # Middle finger
            (self.WRIST, self.MIDDLE_FINGER_MCP),
            (self.MIDDLE_FINGER_MCP, self.MIDDLE_FINGER_PIP),
            (self.MIDDLE_FINGER_PIP, self.MIDDLE_FINGER_DIP),
            (self.MIDDLE_FINGER_DIP, self.MIDDLE_FINGER_TIP),
            # Ring finger
            (self.WRIST, self.RING_FINGER_MCP),
            (self.RING_FINGER_MCP, self.RING_FINGER_PIP),
            (self.RING_FINGER_PIP, self.RING_FINGER_DIP),
            (self.RING_FINGER_DIP, self.RING_FINGER_TIP),
            # Pinky
            (self.WRIST, self.PINKY_MCP),
            (self.PINKY_MCP, self.PINKY_PIP),
            (self.PINKY_PIP, self.PINKY_DIP),
            (self.PINKY_DIP, self.PINKY_TIP),
            # Palm (MCP connections)
            (self.INDEX_FINGER_MCP, self.MIDDLE_FINGER_MCP),
            (self.MIDDLE_FINGER_MCP, self.RING_FINGER_MCP),
            (self.RING_FINGER_MCP, self.PINKY_MCP),
        ]
        
        lines = []
        points = []
        
        for start_idx, end_idx in connections:
            start_lm = self.last_landmarks[start_idx]
            end_lm = self.last_landmarks[end_idx]
            
            start_pt = (int(start_lm.x * target_width), int(start_lm.y * target_height))
            end_pt = (int(end_lm.x * target_width), int(end_lm.y * target_height))
            
            lines.append((start_pt, end_pt))
        
        # Landmark-Punkte für Kreise
        for landmark in self.last_landmarks:
            x = int(landmark.x * target_width)
            y = int(landmark.y * target_height)
            points.append((x, y))
        
        return lines, points
    
    def _detect_gesture(self, landmarks):
        """
        Detects if index finger is extended (for cutting) or fist.
        
        Returns:
            str: 'point' (index finger), 'fist' or 'unknown'
        """
        # Check if index finger is extended
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        index_pip = landmarks[self.INDEX_FINGER_PIP]
        index_extended = index_tip.y < index_pip.y
        
        # Check if other fingers are folded
        middle_tip = landmarks[self.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.MIDDLE_FINGER_PIP]
        middle_extended = middle_tip.y < middle_pip.y
        
        ring_tip = landmarks[self.RING_FINGER_TIP]
        ring_pip = landmarks[self.RING_FINGER_PIP]
        ring_extended = ring_tip.y < ring_pip.y
        
        pinky_tip = landmarks[self.PINKY_TIP]
        pinky_pip = landmarks[self.PINKY_PIP]
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        # Pointing gesture: only index finger extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'point'
        
        # Fist: no finger extended
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'fist'
        
        return 'unknown'
    
    def get_frame_with_landmarks(self):
        """
        Returns the current frame with drawn hand landmarks.
        Useful for debugging.
        """
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks manually
                h, w, _ = frame.shape
                for i, landmark in enumerate(hand_landmarks):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                for start, end in connections:
                    start_pt = (int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h))
                    end_pt = (int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
        
        return frame


def test_hand_tracker():
    """Test function for the Hand Tracker"""
    print("Starting Hand Tracker Test...")
    print("Press 'q' to exit")
    print("")
    print("Gestures:")
    print("  - Pointing finger = 'point' (for cutting)")
    print("  - Fist = 'fist' (for pushing)")
    print("")
    
    tracker = HandTracker()
    
    if not tracker.start_camera():
        print("Error: Camera could not be started!")
        return
    
    try:
        while True:
            # Get frame with landmarks
            frame = tracker.get_frame_with_landmarks()
            
            if frame is None:
                continue
            
            # Get hand position and gesture
            result = tracker.get_hand_position()
            
            if result:
                x, y, gesture = result
                
                # Display info on frame
                text = f"Position: ({x:.2f}, {y:.2f}) Gesture: {gesture}"
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Colored circle based on gesture
                color = (0, 255, 0) if gesture == 'point' else (0, 0, 255) if gesture == 'fist' else (255, 255, 0)
                center = (int(x * frame.shape[1]), int(y * frame.shape[0]))
                cv2.circle(frame, center, 20, color, -1)
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Hand Tracker Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        tracker.stop_camera()
        cv2.destroyAllWindows()
        print("Test finished.")


if __name__ == "__main__":
    test_hand_tracker()
