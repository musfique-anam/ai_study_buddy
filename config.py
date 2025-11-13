# config.py
IMG_SIZE = (224, 224) # Default, but FER model might need 48x48
CLASS_NAMES = ["focused", "distracted", "drowsy"]

# thresholds (tweak for your environment)
EYE_AR_THRESH = 0.22   # eye aspect ratio below => likely closed
EYE_AR_CONSEC_FRAMES = 6  # consecutive frames below => drowsy
MOUTH_AR_THRESH = 0.6  # mouth aspect ratio above => possible yawn

# --- NEW FOR PRO MODE ---
# Emotions from FER2013 model
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Which emotions to count as "distracted"
DISTRACTED_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad", "Surprise"}
FOCUSED_EMOTIONS = {"Happy", "Neutral"}