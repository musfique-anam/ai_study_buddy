# config.py
IMG_SIZE = (224, 224) 
CLASS_NAMES = ["focused", "distracted", "drowsy"]

# thresholds 
EYE_AR_THRESH = 0.22   # likely closed
EYE_AR_CONSEC_FRAMES = 6  # drowsy
MOUTH_AR_THRESH = 0.6  # yawn

# Emotions from FER2013 model
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Which emotions to count as "distracted"
DISTRACTED_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad", "Surprise"}
FOCUSED_EMOTIONS = {"Happy", "Neutral"}