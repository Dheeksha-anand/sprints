import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
from threading import Thread
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Session state
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""
if 'last_detected_sign' not in st.session_state:
    st.session_state.last_detected_sign = ""
if 'last_sign_time' not in st.session_state:
    st.session_state.last_sign_time = time.time()
if 'pending_signs' not in st.session_state:
    st.session_state.pending_signs = []

def speak_text(text):
    # Split text into words
    words = text.strip().split()
    
    for word in words:
        if os.name == 'nt':
            os.system(f'powershell -c "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{word}\')"')
        elif os.name == 'posix':
            os.system(f'espeak "{word}"')
        elif os.name == 'darwin':
            os.system(f'say "{word}"')
        # Wait for 3 seconds between words
        time.sleep(3)

def get_finger_angles(landmarks):
    angles = {}
    # Calculate angles for all fingers
    for finger in range(5):
        base = finger * 4 + 1
        angles[finger] = np.degrees(np.arctan2(
            landmarks[base + 3].y - landmarks[base].y,
            landmarks[base + 3].x - landmarks[base].x
        ))
    return angles

def detect_sign(landmarks, image_height, image_width):
    def is_finger_extended(tip_id, pip_id, threshold=0.05):
        return landmarks[tip_id].y < landmarks[pip_id].y - threshold

    def get_distance(p1_id, p2_id):
        return np.sqrt(
            (landmarks[p1_id].x - landmarks[p2_id].x) ** 2 +
            (landmarks[p1_id].y - landmarks[p2_id].y) ** 2
        )

    # Get finger states
    thumb_extended = landmarks[4].x > landmarks[3].x
    index_extended = is_finger_extended(8, 6)
    middle_extended = is_finger_extended(12, 10)
    ring_extended = is_finger_extended(16, 14)
    pinky_extended = is_finger_extended(20, 18)
    
    fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    angles = get_finger_angles(landmarks)

    # A: Fist with thumb to the side
    if not any([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_extended:
        return "A", (0, 255, 0)

    # B: All fingers straight up, thumb tucked
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "B", (0, 255, 0)

    # C: Curved hand, fingers together
    elif all([not index_extended, not middle_extended, not ring_extended, not pinky_extended]) and \
         get_distance(8, 12) < 0.05 and landmarks[8].x > landmarks[7].x:
        return "C", (0, 255, 0)

    # D: Index up, others curled
    elif index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "D", (0, 255, 0)

    # E: All fingers curled
    elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "E", (0, 255, 0)

    # F: Index and thumb touching, others extended
    elif not index_extended and all([middle_extended, ring_extended, pinky_extended]):
        return "F", (0, 255, 0)

    # G: Index pointing to side, thumb extended
    elif index_extended and thumb_extended and landmarks[8].x > landmarks[7].x:
        return "G", (0, 255, 0)

    # H: Index and middle side by side
    elif index_extended and middle_extended and not any([ring_extended, pinky_extended]) and \
         abs(landmarks[8].y - landmarks[12].y) < 0.05:
        return "H", (0, 255, 0)

    # I: Pinky up only
    elif pinky_extended and not any([thumb_extended, index_extended, middle_extended, ring_extended]):
        return "I", (0, 255, 0)

    # J: Pinky up and moving (would need motion tracking)
    elif pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "J", (0, 255, 0)

    # K: Index and middle up in V, thumb between
    elif index_extended and middle_extended and thumb_extended and \
         not any([ring_extended, pinky_extended]):
        return "K", (0, 255, 0)

    # L: L shape with index and thumb
    elif index_extended and thumb_extended and \
         not any([middle_extended, ring_extended, pinky_extended]) and \
         landmarks[4].y > landmarks[8].y:
        return "L", (0, 255, 0)

    # M: Three fingers over thumb
    elif not any([thumb_extended, ring_extended, pinky_extended]) and \
         all([index_extended, middle_extended]):
        return "M", (0, 255, 0)

    # N: Two fingers over thumb
    elif not any([thumb_extended, middle_extended, ring_extended, pinky_extended]) and \
         index_extended:
        return "N", (0, 255, 0)

    # O: Fingers curved into O shape
    elif get_distance(4, 8) < 0.1 and not any([middle_extended, ring_extended, pinky_extended]):
        return "O", (0, 255, 0)

    # P: Index down, thumb out
    elif not any([index_extended, middle_extended, ring_extended]) and thumb_extended:
        return "P", (0, 255, 0)

    # R: Crossed fingers
    elif index_extended and middle_extended and \
         not any([ring_extended, pinky_extended]) and \
         landmarks[8].x > landmarks[12].x:
        return "R", (0, 255, 0)

    # S: Fist with thumb over fingers
    elif not any(fingers_extended) and landmarks[4].x < landmarks[3].x:
        return "S", (0, 255, 0)

    # T: Index bent, thumb between
    elif not any([index_extended, middle_extended, ring_extended, pinky_extended]) and \
         thumb_extended and landmarks[4].y < landmarks[8].y:
        return "T", (0, 255, 0)

    # U: Index and middle parallel up
    elif index_extended and middle_extended and \
         not any([ring_extended, pinky_extended]) and \
         abs(landmarks[8].x - landmarks[12].x) < 0.05:
        return "U", (0, 255, 0)

    # V: Index and middle in V shape
    elif index_extended and middle_extended and \
         not any([ring_extended, pinky_extended]) and \
         landmarks[8].x - landmarks[12].x > 0.1:
        return "V", (0, 255, 0)

    # W: Index, middle, and ring fingers spread
    elif index_extended and middle_extended and ring_extended and \
         not pinky_extended:
        return "W", (0, 255, 0)

    # X: Index hook
    elif not index_extended and not any([middle_extended, ring_extended, pinky_extended]) and \
         landmarks[8].y > landmarks[7].y:
        return "X", (0, 255, 0)

    # Y: Thumb and pinky extended
    elif thumb_extended and pinky_extended and \
         not any([index_extended, middle_extended, ring_extended]):
        return "Y", (0, 255, 0)

    # Z: Index drawing Z (would need motion tracking)
    elif index_extended and not any([middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Z", (0, 255, 0)

    return "", (128, 128, 128)

# Streamlit UI
st.title('ASL Alphabet Detection with Voice Output')
st.sidebar.title('Controls')

st.sidebar.markdown("""
### 
""")

# Add voice output button
voice_button = st.sidebar.button('Speak Text (3s pause between words)')
clear_button = st.sidebar.button('Clear Text')
space_button = st.sidebar.button('Add Space')

if clear_button:
    st.session_state.current_sentence = ""
    st.session_state.last_detected_sign = ""
    st.session_state.last_sign_time = time.time()
    st.session_state.pending_signs = []

if space_button:
    st.session_state.current_sentence += " "

if voice_button and st.session_state.current_sentence.strip():
    thread = Thread(target=speak_text, args=(st.session_state.current_sentence.strip(),))
    thread.start()

st.markdown('## Output')
gesture_text = st.empty()
sentence_display = st.empty()
stframe = st.empty()

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        image_height, image_width, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                sign, color = detect_sign(hand_landmarks.landmark, image_height, image_width)

                if sign and (time.time() - st.session_state.last_sign_time > 1.5):
                    st.session_state.current_sentence += sign
                    st.session_state.last_detected_sign = sign
                    st.session_state.last_sign_time = time.time()
                    st.session_state.pending_signs.append(sign)

                cv2.putText(frame, f"Letter: {sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                gesture_text.markdown(f"## Detected Letter: {sign}")

        sentence_display.markdown(f"### Text: {st.session_state.current_sentence.strip()}")
        stframe.image(frame, channels='BGR', use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()