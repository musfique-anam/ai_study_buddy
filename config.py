# config.py
IMG_SIZE = (224, 224)
CLASS_NAMES = ["focused", "distracted", "drowsy"]

# thresholds (tweak for your environment)
EYE_AR_THRESH = 0.22   # eye aspect ratio below => likely closed
EYE_AR_CONSEC_FRAMES = 6  # consecutive frames below => drowsy
MOUTH_AR_THRESH = 0.6  # mouth aspect ratio above => possible yawn
DISTRACTED_EMOTIONS = {"angry", "disgust", "fear", "surprise"}  # treat as distracted
FOCUSED_EMOTIONS = {"neutral", "happy"}  # emphasize as focused
