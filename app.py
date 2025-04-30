import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# Page config
st.set_page_config(page_title="MFCC Audio Analysis", layout="wide")

# Title and Sidebar Info
st.title("DSP LAB END TERM EXAMINATION")
st.markdown("#####  Submitted by: **Harish Kushwaha** | Scholar No: `2311401104`")

run_id = uuid.uuid4()
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.sidebar.title("📋 Session Info")
st.sidebar.write(f"🆔 Run ID:\n`{run_id}`")
st.sidebar.write(f"🕒 Timestamp:\n`{run_time}`")

# Controls in Sidebar
st.sidebar.title("🎛️ Controls")
frame_ms = st.sidebar.slider("Frame size (ms)", 10, 50, 25, 5)
overlap_percent = st.sidebar.slider("Overlap (%)", 0, 90, 50, 5)
num_mels = st.sidebar.selectbox("Mel Filter Bank Size", [20, 30, 40], index=1)
new_sr = st.sidebar.selectbox("Resample to (Hz)", [8000, 16000, 22050], index=1)

# File Uploader
st.markdown("#### 🎙️ Upload a `.wav` file to begin analysis:")
uploaded_file = st.file_uploader("", type="wav")

# Stylish Buttons in One Row
st.markdown("---")
st.markdown("### 🧪 Choose an Analysis to Perform")

col1, col2, col3, col4 = st.columns(4)
with col1:
    btn1 = st.button("📊 Spectrogram vs MFCC", use_container_width=True)
with col2:
    btn2 = st.button("🎛️ Frame & Overlap", use_container_width=True)
with col3:
    btn3 = st.button("🎚️ Mel Filter Custom", use_container_width=True)
with col4:
    btn4 = st.button("🔄 Resample & Degrade", use_container_width=True)

st.markdown("---")

# Main Logic
if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)

    if btn1:
        st.subheader("🔍 Spectrogram vs MFCC Interpretation")
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))

        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=axs[0])
        axs[0].set_title("Spectrogram")
        librosa.display.specshow(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), sr=sr, x_axis='time', ax=axs[1])
        axs[1].set_title("MFCCs")

        st.pyplot(fig)

    elif btn2:
        st.subheader("🛠️ Frame Size and Overlap Effect on MFCC")
        frame_len = int(sr * (frame_ms / 1000))
        hop_len = int(frame_len * (1 - overlap_percent / 100))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
        ax.set_title(f"MFCCs - Frame: {frame_ms}ms | Overlap: {overlap_percent}%")
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

    elif btn3:
        st.subheader("🎚️ Mel Filter Bank Customization")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=axs[0])
        axs[0].set_title(f"Mel Filter Bank ({num_mels} Filters)")

        librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axs[1])
        axs[1].set_title("MFCCs")
        st.pyplot(fig)

    elif btn4:
        st.subheader("🔄 Sampling Rate Adjustment & MFCC Degradation")
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=new_sr)

        fig, axs = plt.subplots(1, 3, figsize=(18, 4))

        axs[0].plot(np.linspace(0, len(y_resampled) / new_sr, len(y_resampled)), y_resampled)
        axs[0].set_title("Time Domain")

        S = librosa.amplitude_to_db(np.abs(librosa.stft(y_resampled)), ref=np.max)
        librosa.display.specshow(S, sr=new_sr, x_axis='time', y_axis='log', ax=axs[1])
        axs[1].set_title("Spectrogram")

        mfccs = librosa.feature.mfcc(y=y_resampled, sr=new_sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', sr=new_sr, ax=axs[2])
        axs[2].set_title("MFCCs")

        st.pyplot(fig)
