# pro_detector.py
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
from imutils import face_utils
from config import (
    EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MOUTH_AR_THRESH,
    EMOTION_CLASSES, DISTRACTED_EMOTIONS
)

class ProEstimator:
    def __init__(self):
        # 1. Dlib Heuristics (like FocusEstimator)
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # 2. Emotion DL Model (Keras)
        # --- !! IMPORTANT: Place your fer_model.h5 file in your project folder !! ---
        self.emotion_model = load_model("fer_model.h5") 
        
        # 3. Face detector for Emotion (Haar Cascade is faster for this)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 4. State variables
        self.eye_counter = 0
        self.blink_alert = False
        self.yawn_alert = False
        self.posture_alert = False
        self.emotion_alert = False
        self.current_emotion = "---"

        # 5. Counters
        self.focused_seconds = 0
        self.drowsy_seconds = 0
        self.distracted_seconds = 0

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1]-eye[5])
        B = np.linalg.norm(eye[2]-eye[4])
        C = np.linalg.norm(eye[0]-eye[3])
        if C == 0: return 0.3
        return (A+B)/(2.0*C)

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2]-mouth[10])
        B = np.linalg.norm(mouth[4]-mouth[8])
        C = np.linalg.norm(mouth[0]-mouth[6])
        if C == 0: return 0.0
        return (A+B)/(2.0*C)

    def process_frame(self, frame):
        FRAME_TIME_DELTA = 1/30.0 # Assume ~30FPS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        is_drowsy = False
        is_distracted_by_posture_or_yawn = False
        is_distracted_by_emotion = False
        
        # --- 1. Dlib Landmark Detection (for Drowsy/Yawn) ---
        dlib_faces = self.dlib_detector(gray, 0)
        
        if len(dlib_faces) == 0:
            is_distracted_by_posture_or_yawn = True # No face, bad posture
            self.posture_alert = True
        else:
            face = dlib_faces[0]
            shape = self.dlib_predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            # Eye logic (Drowsy)
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
            if ear < EYE_AR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                    is_drowsy = True
                    self.blink_alert = True
            else:
                self.eye_counter = 0
                self.blink_alert = False

            # Mouth logic (Yawn)
            mouth = shape_np[self.mStart:self.mEnd]
            mar = self.mouth_aspect_ratio(mouth)
            self.yawn_alert = mar > MOUTH_AR_THRESH
            if self.yawn_alert:
                is_distracted_by_posture_or_yawn = True

            # Posture logic
            nose = shape_np[30]
            chin = shape_np[8]
            posture_deviation = abs(nose[0] - chin[0])
            self.posture_alert = (posture_deviation > 40 or nose[1] > chin[1])
            if self.posture_alert:
                is_distracted_by_posture_or_yawn = True

            # Draw dlib landmarks
            for (x,y) in np.concatenate([leftEye, rightEye, mouth]):
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        # --- 2. Emotion Detection (for Distraction) ---
        haar_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(haar_faces) == 0:
            self.current_emotion = "---"
            self.emotion_alert = False
        else:
            (x, y, w, h) = haar_faces[0] # Get first face
            
            # Crop and prepare face for FER model
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Normalize and expand dims for model
            face_pixels = face_roi_resized.astype('float32') / 255.0
            face_pixels = np.expand_dims(face_pixels, axis=0)
            face_pixels = np.expand_dims(face_pixels, axis=-1)
            
            # Predict emotion
            predictions = self.emotion_model.predict(face_pixels)
            emotion_idx = np.argmax(predictions[0])
            self.current_emotion = EMOTION_CLASSES[emotion_idx]
            
            # Check if emotion is a distraction
            if self.current_emotion in DISTRACTED_EMOTIONS:
                is_distracted_by_emotion = True
                self.emotion_alert = True
                color = (0, 0, 255) # Red
            else:
                is_distracted_by_emotion = False
                self.emotion_alert = False
                color = (255, 0, 0) # Blue
            
            # Draw emotion box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, self.current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- 3. Update Counters ---
        is_distracted = is_distracted_by_posture_or_yawn or is_distracted_by_emotion
        
        if is_drowsy:
            self.drowsy_seconds += FRAME_TIME_DELTA
        elif is_distracted:
            self.distracted_seconds += FRAME_TIME_DELTA
        else:
            self.focused_seconds += FRAME_TIME_DELTA

        return frame