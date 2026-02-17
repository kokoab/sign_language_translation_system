import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from src.model import SLTModel
from src.utils import get_hand_adj

# Load Config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load('weights/slt_model.pth')
labels = checkpoint['labels']
adj = get_hand_adj().to(device)
model = SLTModel(num_classes=len(labels), adj=adj).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, model_complexity=1)

sequence = []
sentence = []
current_label = ""
current_conf = 0.0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    image = cv2.cvtColor(frame, cv2.BGR2RGB)
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        points = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
        sequence.append(points)
    else:
        sequence.append(np.zeros(63))

    sequence = sequence[-60:]

    if len(sequence) == 60:
        input_data = np.array(sequence).reshape(60, 21, 3).astype(np.float32)
        
        # MUST MATCH TRAINING PREPROCESSING
        wrist = input_data[:, 0, :]
        input_data = input_data - wrist[:, np.newaxis, :]
        max_dist = np.max(np.linalg.norm(input_data, axis=2))
        if max_dist > 0:
            input_data = input_data / max_dist
        
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            res = model(input_tensor)
            prob = F.softmax(res, dim=1)
            conf, idx = torch.max(prob, 1)
            current_conf = conf.item()
            current_label = labels[idx.item()]

            if current_conf > 0.95:
                if not sentence or current_label != sentence[-1]:
                    sentence.append(current_label)

    # UI Overlay
    cv2.putText(frame, f"Pred: {current_label} ({current_conf*100:.1f}%)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {''.join(sentence)}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Improved ASL SLT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()