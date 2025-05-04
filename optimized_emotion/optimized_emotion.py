
import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import joblib
import time
import glob
import pandas as pd

model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")
emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']

audio_folder = "audio_samples"
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

results = []

for audio_file in audio_files:
    start_time = time.time()

    try:
        sr, audio = wav.read(audio_file)
    except Exception as e:
        results.append({
            "file": os.path.basename(audio_file),
            "emotion": f"Error: {str(e)}",
            "time_sec": 0
        })
        continue

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    mfcc_features = mfcc(audio, samplerate=sr, numcep=13, winlen=0.03, winstep=0.02)

    mfcc_features = scaler.transform(mfcc_features)

    features = np.mean(mfcc_features, axis=0).reshape(1, -1)

    prediction = model.predict(features)
    predicted_emotion = emotion_labels[int(prediction[0])]

    elapsed = round(time.time() - start_time, 3)

    results.append({
        "file": os.path.basename(audio_file),
        "emotion": predicted_emotion,
        "time_sec": elapsed
    })

df = pd.DataFrame(results)
df.to_csv("emotion_results.csv", index=False)
print(df)
