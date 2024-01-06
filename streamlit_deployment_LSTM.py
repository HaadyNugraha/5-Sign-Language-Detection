import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

#Loading Model
model = load_model('action_LSTM.h5')

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Defining actions
actions = np.array(['Saya', 'Anda', 'Terima Kasih', 'Maaf', 'Cinta'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (120, 127, 245), (130, 125, 145)]

sequence = []

#Extracting Keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#Vizualizing Probabilities
def prob_viz(res, actions, frame, colors):
    # Create a bar chart with probabilities for each action
    prob_chart = st.bar_chart({action: [prob] for action, prob in zip(actions, res)}, color=colors)
    return prob_chart

#Real Time Prediction
def predict_action(frame):
    global sequence  # Use the global sequence variable

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Extracting Keypoints
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-20:]

    if len(sequence) == 20:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        action = actions[np.argmax(res)]
        return action, prob_viz(res, actions, frame, colors)

    return None, frame

#For Streamlit
def main():
    st.title("Action Recognition using Streamlit")

    video_feed = st.empty()

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()

        action, viz_frame = predict_action(frame)

        if action:
            st.write("Recognized Action:", action)

        if isinstance(viz_frame, np.ndarray):
            viz_frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)

            video_feed.image(viz_frame_rgb, channels="RGB", use_column_width=True)

        #Stop Button
        stop_button_key = "stop_button_" + str(time.time())

        if st.button("Stop", key=stop_button_key):
            break

    cap.release()
    st.warning("Streamlit app has been stopped.")

if __name__ == "__main__":
    main()