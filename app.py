import streamlit as st

# Data manipulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Feature extraction
import scipy
import librosa
#import python_speech_features as mfcc
import os
from scipy.io.wavfile import read
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Model training
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import pickle

# Live recording
import sounddevice as sd
import soundfile as sf

st.title("Speech Emotion Detection")
st.write("This application demonstrates a simple Voice Gender Detection. Voice gender identification relies on three important steps.")
st.write("- Extracting from the training set MFCC features (13 usually) for each gender")
st.write("- Train a GMM on those features")
st.write("- In prediction, compute the likelihood of each gender using the trained GMM, and pick the most likely gender")


st.subheader("Ready to try it on your voice?")

st.sidebar.title("Parameters")
duration = st.sidebar.slider("Recording duration", 0.0, 10.0, 3.0)

def chromogram_calculate(recording):
    chrom_cols = []
    #create columns
    for i in range(1,13):
        chrom_cols.append("CHROM"+str(i))
        
    # Extract MFCC features and create a data frame    
    chromogram = np.mean(librosa.feature.chroma_stft(recording),axis=1)
    chromogramr = chromogram.reshape(-1,12)
    df = pd.DataFrame(chromogramr, columns=chrom_cols)
    return(df)


# Function to calculate MelSpectrogram features for each trimmed audio signal
def mel_calculate(recording, n=10):
    mel=[]
    mel_cols = []
    #create columns
    for i in range(1,n+1):
        mel_cols.append("MEL"+str(i))
        
    # Extract MFCC features and create a data frame    
    #print(np.mean(librosa.feature.melspectrogram(audio,n_mels=n),axis=1))
    mel = np.mean(librosa.feature.melspectrogram(recording,n_mels=n),axis=1)
    melr = mel.reshape(-1,n)
    df = pd.DataFrame(melr, columns=mel_cols)
    return(df)

def calculate_BER(audio, sr=22050, split_frequency = 2000): # deafult sample rate to 22050
    # Calculate frequency bin
    stft = librosa.stft(recording)
    frequency_range = sr/2
    frequency_bins = frequency_range/stft.shape[0]
    split_frequency_bin = math.floor(split_frequency/frequency_bins)
        
    # calculate power spectrogram
    power_spectrogram = np.abs(stft) ** 2
    
    # calculate BER value for each frame
    sum_low_frequencies = power_spectrogram[:split_frequency_bin].sum()
    sum_high_frequencies = power_spectrogram[split_frequency_bin:].sum()
    ber_audio = sum_low_frequencies / sum_high_frequencies
    ber = np.mean(ber_audio)
    return(ber)
def mfcc_calculate(audio, n=13):
    #create columns
    for i in range(1,n+1):
        mfcc_cols.append("MFCC"+str(i))
        
    # Extract MFCC features and create a data frame    
    mfcc = np.mean(librosa.feature.mfcc(recording,n_mfcc=n),axis=1)
    
    df = pd.DataFrame(mfcc, columns=mfcc_cols)
    return(df)



def record_and_predict( sr=16000, channels=2, duration=5, filename='pred_record.wav'):
    """
    Records live voice and returns the identified gender
    """ 
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels,dtype='float64').reshape(-1)
    sd.wait()
    df = mel_calculate(recording,512)
    mel_cols = df.columns
    rms = np.mean(librosa.feature.rms(recording, frame_length=2048,hop_length=512))
    df['RMS'] = rms
    zcr = np.mean(librosa.feature.zero_crossing_rate(recording, frame_length=2048,hop_length=512))
    df['ZCR'] = zcr 
    final_df  = df 
    df = chromogram_calculate(recording)
    final_df = pd.concat([final_df,df],axis=1)
    spetral_centroid = np.mean(librosa.feature.spectral_centroid(recording, n_fft=2048,hop_length=512))
    final_df['Spectral Centroid'] = spetral_centroid
    BER = calculate_BER(recording)
    final_df['BER'] = BER
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(recording, n_fft=2048,hop_length=512))
    final_df['Bandwidth'] = bandwidth
    df = mfcc_calculate(recording, 40)
    final_df  = pd.concat([final_df,df],axis=1)
    scaled_data = pd.DataFrame(scale(final_df))
    lda = LinearDiscriminantAnalysis()
    transformed_data = lda.fit_transform(scaled_data)
    transformed_data.shape
    pd.DataFrame(transformed_data)
    nb = joblib.load('naive_bayes.pkl')
    y_predicted = nb.predict(transformed_data)
    return(y_predicted)


if st.button("Start Recording"):
    with st.spinner("Recording..."):
        gender = record_and_predict(duration=duration)
        st.write("The identified gender is: " + gender)
