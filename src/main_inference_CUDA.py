import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Use MPS for M4 MacBook Air, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 1. MODEL DEFINITION (Must match Training Logic) ---
class SLTModel(torch.nn.Module):
    def __init__(self, num_classes, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn_spatial = torch.nn.Linear(3, 128)
        self.ln1 = torch.nn.LayerNorm(128)
        self.projection = torch.nn.Linear(21 * 128, 256)
        self.geom_fc = torch.nn.Linear(10, 64)
        self.d_model = 320
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, 30, self.d_model))
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4), 
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, f, j, c = x.shape
        def dist(p1, p2): return torch.sqrt(torch.sum((p1 - p2)**2, dim=-1) + 1e-8)
        
        # Enhanced Features for R/V/K/S/T/M/N
        geom_emb = F.relu(self.geom_fc(torch.stack([
            dist(x[:,:,4], x[:,:,8]),   # Thumb-Index
            dist(x[:,:,8], x[:,:,12]),  # Index-Middle (V vs R)
            dist(x[:,:,12], x[:,:,16]), # Middle-Ring
            dist(x[:,:,16], x[:,:,20]), # Ring-Pinky
            dist(x[:,:,4], x[:,:,12]),  # Thumb-Middle (K)
            dist(x[:,:,4], x[:,:,16]),  # Thumb-Ring
            dist(x[:,:,4], x[:,:,20]),  # Thumb-Pinky
            dist(x[:,:,0], x[:,:,4]),   # Thumb Extension
            dist(x[:,:,0], x[:,:,8]),   # Index Extension
            dist(x[:,:,0], x[:,:,12])   # Middle Extension
        ], dim=-1)))
        
        x = torch.matmul(self.adj, x) 
        x = self.gcn_spatial(x)
        x = self.ln1(x).view(b, f, -1)
        x = F.relu(self.projection(x))
        x = torch.cat([x, geom_emb], dim=-1) + self.pos_encoder
        return self.classifier(self.transformer(x).mean(dim=1))

# --- 2. SETUP & WEIGHT LOADING ---
WINDOW_SIZE = 30 
adj = torch.eye(21)
conns = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20)]
for i, j in conns: adj[i,j]=1; adj[j,i]=1

# Ensure path is correct for your Mac folder structure
try:
    checkpoint = torch.load('weights/slt_model.pth', map_location=device)
    labels = checkpoint['labels']
    model = SLTModel(num_classes=len(labels), adj=adj).to(device)
    model.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 3. RECOGNITION VARIABLES ---
sequence = []
sentence = []
stability_buffer = [] 
STABILITY_THRESHOLD = 3 # Confirms after 3 matching frames
last_confirmed_char = ""

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity=1)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # Scoped variables to prevent NameError
    current_char = "Waiting..."
    current_conf = 0.0

    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        
        # Capture landmarks
        pts = np.array([[l.x, l.y, l.z] for l in res.multi_hand_landmarks[0].landmark]).flatten()
        sequence.append(pts)
    else:
        # Faster clear when hand is missing to reset logic
        sequence = sequence[5:] if len(sequence) > 5 else []
        stability_buffer = []

    if len(sequence) >= WINDOW_SIZE:
        sequence = sequence[-WINDOW_SIZE:]
        input_data = np.array(sequence).reshape(WINDOW_SIZE, 21, 3).astype(np.float32)
        
        # --- NEW BONE-LENGTH NORMALIZATION ---
        # Center on wrist
        wrist = input_data[:, 0, np.newaxis, :]
        input_data -= wrist
        
        # Scale by reference bone (Middle Finger MCP - Landmark 9)
        # This makes the hand size invariant to camera distance
        ref_dist = np.mean(np.linalg.norm(input_data[:, 9, :], axis=-1))
        if ref_dist > 0:
            input_data /= ref_dist
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)
            out = model(input_tensor)
            prob = F.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)
            
            current_char = labels[idx.item()]
            current_conf = conf.item()

            # Stability logic
            if current_conf > 0.80:
                stability_buffer.append(current_char)
            else:
                stability_buffer.append("None")
            
            stability_buffer = stability_buffer[-STABILITY_THRESHOLD:]

            if len(stability_buffer) == STABILITY_THRESHOLD and all(x == current_char for x in stability_buffer):
                if current_char not in ["Idle", "None"] and current_char != last_confirmed_char:
                    sentence.append(current_char)
                    last_confirmed_char = current_char
                
                if current_char == "Idle":
                    last_confirmed_char = ""

    # --- UI RENDERING ---
    cv2.rectangle(frame, (0,0), (frame.shape[1], 80), (30, 30, 30), -1)
    
    # Text feedback
    display_color = (0, 255, 0) if current_conf > 0.8 else (0, 165, 255)
    cv2.putText(frame, f"PRED: {current_char} ({current_conf*100:.0f}%)", (20, 30), 2, 0.7, display_color, 2)
    cv2.putText(frame, f"TEXT: {''.join(sentence)}", (20, 65), 2, 0.9, (255, 255, 255), 2)
    
    cv2.imshow('Rapid SLT - Bone Normalized', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): 
        sentence = []
        last_confirmed_char = ""

cap.release()
cv2.destroyAllWindows()