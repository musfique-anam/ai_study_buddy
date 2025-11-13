import dlib

DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def get_dlib_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    return detector, predictor
