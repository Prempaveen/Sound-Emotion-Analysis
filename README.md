# Sound-Emotion-Analysis


## Emotion Detection from Audio

This project detects the **emotion of audio files** (Happy, Sad, Angry, Neutral) using MFCC features and a pre-trained SVM model. The code is optimized for fast batch processing and suitable for performance evaluation in Operating Systems projects.

---

## Folder Structure

```
emotion_project/
├── dataset/                     # For training (grouped by emotion)
│   ├── Happy/
│   ├── Sad/
│   ├── Angry/
│   ├── Neutral/
│
├── audio_samples/               # For prediction (input audio files)
│   ├── test1.wav
│   ├── test2.wav
│
├── train_model.py               # Script to train SVM + scaler
├── optimized_emotion.py         # Optimized inference script
├── svm_model.joblib             # Saved model (generated after training)
├── scaler.joblib                # Saved scaler (generated after training)
├── emotion_results.csv          # Output results (generated after prediction)
└── README.md                    # Instructions (this file)
```

---

## Step 1: Train the Model

Prepare your dataset with `.wav` files in the `dataset/` folder, grouped by emotion names (Happy, Sad, Angry, Neutral).  
Then run:

```bash
pip install python_speech_features scikit-learn joblib
python train_model.py
```

This will create `svm_model.joblib` and `scaler.joblib`.

---

## Step 2: Predict Emotion from Audio Files

Place the `.wav` files you want to analyze into the `audio_samples/` folder.  
Then run:

```bash
python optimized_emotion.py
```

This will create a `emotion_results.csv` file with the following columns:

- `file` – the name of the audio file
- `emotion` – predicted emotion label
- `time_sec` – time taken to process the file

---

## Dependencies

Install the required libraries using pip:

```bash
pip install python_speech_features scikit-learn joblib pandas
```

---

## Notes

- Audio files should be in `.wav` format.
- The MFCC uses 13 coefficients, 30ms window, and 20ms hop step.
- Stereo audio will be converted to mono automatically.

---

## Contact

If you have any questions, feel free to ask!
