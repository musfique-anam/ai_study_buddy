# focus_detector.py
import cv2
import numpy as np
from model_utils import get_dlib_detector
from imutils import face_utils
# --- IMPORTED FROM CONFIG ---
from config import EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MOUTH_AR_THRESH

class FocusEstimator:
    def __init__(self):
        self.detector, self.predictor = get_dlib_detector()
        self.eye_counter = 0
        self.blink_alert = False
        self.yawn_alert = False
        self.posture_alert = False
        self.focused_seconds = 0
        self.drowsy_seconds = 0
        self.distracted_seconds = 0 # <-- Includes the fix from our last conversation

        # Get face landmark indices
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1]-eye[5])
        B = np.linalg.norm(eye[2]-eye[4])
        C = np.linalg.norm(eye[0]-eye[3])
        if C == 0: return 0.3 # Avoid division by zero
        return (A+B)/(2.0*C)

    def mouth_aspect_ratio(self, mouth):
        # vertical distances
        A = np.linalg.norm(mouth[2]-mouth[10]) # 51-59
        B = np.linalg.norm(mouth[4]-mouth[8]) # 53-57
        # horizontal distance
        C = np.linalg.norm(mouth[0]-mouth[6]) # 49-55
        if C == 0: return 0.0 # Avoid division by zero
        return (A+B)/(2.0*C)

    def process_frame(self, frame):
        # --- Estimate frame processing time (assuming ~30FPS) ---
        FRAME_TIME_DELTA = 1/30.0 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        # Frame status
        is_focused = True
        is_drowsy = False
        is_distracted = False

        # --- THIS IS THE CRITICAL FIX ---
        if len(faces) == 0:
            # BUG FIX: If no face is detected, count as distracted
            is_distracted = True 
            self.posture_alert = True 
        else:
            # A face was found, so run the normal logic
            face = faces[0] # Assume one student
            shape = self.predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            # Eyes
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR)/2.0

            if ear < EYE_AR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                    is_drowsy = True
                    self.blink_alert = True
            else:
                self.eye_counter = 0
                self.blink_alert = False

            # Mouth (Yawn)
            mouth = shape_np[self.mStart:self.mEnd]
            mar = self.mouth_aspect_ratio(mouth)
            
            if mar > MOUTH_AR_THRESH:
                is_distracted = True
                self.yawn_alert = True
            else:
                self.yawn_alert = False

            # Posture (Simple check)
            nose = shape_np[30] # Nose tip
            chin = shape_np[8] # Chin bottom
            
            posture_deviation = abs(nose[0] - chin[0])
            if posture_deviation > 40 or nose[1] > chin[1]: # Tweak '40'
                is_distracted = True
                self.posture_alert = True
            else:
                self.posture_alert = False

            # Draw landmarks
            for (x,y) in np.concatenate([leftEye, rightEye, mouth]):
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)

            # Rectangle with color alerts
            x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()
            color = (0,255,0) # Green = Focused
            if is_drowsy:
                color = (0,0,255) # Red = Drowsy
            elif is_distracted:
                color = (0,255,255) # Yellow = Distracted
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        # --- END OF THE FIX ---

        # Update counters
        if is_drowsy:
            self.drowsy_seconds += FRAME_TIME_DELTA
        elif is_distracted:
            self.distracted_seconds += FRAME_TIME_DELTA
        else: # Focused
            self.focused_seconds += FRAME_TIME_DELTA

        return frame