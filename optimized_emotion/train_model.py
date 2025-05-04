
import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Dataset path
dataset_path = "dataset"
emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']

X = []
y = []

print("Extracting features...")

# Loop through each emotion folder
for idx, emotion in enumerate(emotion_labels):
    emotion_folder = os.path.join(dataset_path, emotion)
    if not os.path.exists(emotion_folder):
        continue

    for file in os.listdir(emotion_folder):
        if not file.endswith(".wav"):
            continue

        try:
            sr, audio = wav.read(os.path.join(emotion_folder, file))

            # Stereo to mono
            if len(audio.shape) == 2:
                audio = np.mean(audio, axis=1)

            # Extract MFCC
            mfcc_feat = mfcc(audio, samplerate=sr, numcep=13, winlen=0.03, winstep=0.02)
            mfcc_feat = np.mean(mfcc_feat, axis=0)

            X.append(mfcc_feat)
            y.append(idx)

        except Exception as e:
            print(f"Error processing {file}: {e}")

X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "svm_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Training completed and model saved!")
