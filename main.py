import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read video
cap = cv2.VideoCapture('input.mp4')
flow_magnitudes = []
prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Denoise
    
    if prev_gray is not None:
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        # Calculate magnitude
        dx, dy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(dx**2 + dy**2)
        flow_magnitudes.append(np.mean(magnitude))
    else:
        flow_magnitudes.append(0.0)  # First frame
    
    prev_gray = gray

cap.release()

# Normalize
min_val = min(flow_magnitudes)
max_val = max(flow_magnitudes)
normalized = [(x - min_val) / (max_val - min_val) for x in flow_magnitudes]

# Plot
plt.plot(normalized)
plt.xlabel('Frame Number')
plt.ylabel('Normalized Optic Flow')
plt.title('Optic Flow Over Time')
plt.show()
