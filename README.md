# Project-Voice-Preprocessing
# 🎤 Speaker Identification using Mozilla Common Voice (Urdu)

## 📌 Project Overview
This project implements a complete *speech preprocessing, training, testing, and evaluation pipeline* using the *Mozilla Common Voice Urdu dataset*.  
The objective is to build a *speaker identification system* that classifies an audio sample based on the speaker’s voice characteristics.

The project follows standard *speech signal processing* and *machine learning* practices, making it suitable for academic labs, final-year projects, and research prototypes.

---

## 📂 Dataset
*Mozilla Common Voice – Urdu*

- Audio format: .mp3
- Sampling rate: variable (resampled to 16 kHz)
- Language: Urdu
- Dataset structure:
mozilla-common-voice-urdu/
├── clips/
│ ├── *.mp3
├── train.tsv
├── test.tsv
└── validated.tsv

*Labels used:* client_id (Speaker ID)  
Only the *top N speakers* are selected to ensure balanced training.

---

## 🧠 Problem Definition
> Given a speech audio sample, predict the *speaker identity*.

This is a *multi-class supervised classification* problem.

---

## ⚙️ Preprocessing Pipeline
Each audio file undergoes the following preprocessing steps:

1. Audio loading and resampling (16 kHz)
2. Silence trimming
3. Amplitude normalization
4. Padding or truncation to fixed duration
5. Feature extraction using *MFCCs*
6. Feature reshaping for CNN input

---

## 🧪 Feature Extraction
- *Mel Frequency Cepstral Coefficients (MFCC)*
- Number of MFCCs: 40
- Chosen due to their effectiveness in capturing vocal tract characteristics

---

## 🏗️ Model Architecture
A *Convolutional Neural Network (CNN)* is used for classification:

- Convolution + ReLU
- Max Pooling
- Batch Normalization
- Fully Connected Layers
- Softmax Output

This architecture efficiently captures time–frequency patterns from MFCC features.

---

## 🏋️ Training Configuration
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 10
- Train/Test Split: 80/20

---

## 📊 Evaluation Metrics
Model performance is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

Evaluation is performed only on *test-set classes* to avoid class mismatch errors.

---

## 📈 Results
The trained model successfully identifies speakers from unseen audio samples, producing a meaningful confusion matrix and classification report.

---

## 🧰 Libraries Used
- Python 3.x
- Librosa
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- TQDM

---

## ▶️ How to Run
1. Open the notebook in *Kaggle*
2. Ensure the dataset path is:
/kaggle/input/mozilla-common-voice-urdu
3. Run all cells sequentially
4. View evaluation metrics and confusion matrix at the end

---

## 📌 Notes
- Sentence-level classification is *not suitable* for Common Voice.
- Speaker identification is a *valid and standard* supervised task.
- The pipeline is *Kaggle-safe* and *viva-ready*.

---

## 🚀 Future Improvements
- Data augmentation (noise, pitch shift)
- CNN + LSTM or Transformer models
- Speech-to-text (ASR) using CTC loss
- Word Error Rate (WER) evaluation

---

## 👤 Author

---

## 📜 License
This project is for *educational and research purposes* only.
