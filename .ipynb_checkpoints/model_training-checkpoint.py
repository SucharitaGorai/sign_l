import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ========== 1️⃣ Load & Preprocess Dataset ==========
dataset_path = "gestures/"
labels = ["A", "B", "C", "D", "E"]
img_size = 64

X_data, y_data = [], []

for idx, label in enumerate(labels):
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))  # Resize
            X_data.append(img)
            y_data.append(idx)

# Convert to NumPy arrays
X_data = np.array(X_data).reshape(-1, img_size, img_size, 1) / 255.0  # Normalize
y_data = to_categorical(np.array(y_data), num_classes=len(labels))  # One-hot encode labels

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ========== 2️⃣ Define CNN Model (With Dropout) ==========
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])


# ========== 3️⃣ Compile and Train Model ==========
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# ========== 4️⃣ Evaluate Model & Plot Confusion Matrix ==========
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print Classification Report
print("📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=labels))

# ========== 5️⃣ Save Model ==========
model.save("sign_language_model.h5")
print("✅ Model saved as sign_language_model.h5")

# ========== 6️⃣ Plot Training vs Validation Accuracy ==========
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()



