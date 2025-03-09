import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Define labels
labels = ["A", "B", "C", "D", "E"]
img_size = 64  # Same size as training

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Draw ROI (Region of Interest)
    x, y, w, h = 100, 100, 200, 200  # Fixed bounding box
    roi = gray[y:y+h, x:x+w]
    
    # Preprocess ROI
    roi = cv2.resize(roi, (img_size, img_size))
    roi = roi.reshape(1, img_size, img_size, 1) / 255.0  # Normalize
    
    # Predict gesture
    pred = model.predict(roi)
    class_index = np.argmax(pred)
    text = labels[class_index]
    
    # Display result
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw box
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
