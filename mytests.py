from scipy.io import loadmat
import numpy as np
import librosa
import librosa.display

dir = "/home/boreas/Downloads/DeepLearning/dataset/MFCC/whale_audio/"
y, sr = librosa.load(dir + "train1.wav", sr=None)
mfcc = librosa.feature.mfcc(y,sr=sr)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=300, hop_length=300)
librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                         x_axis='time', y_axis='mel')
print(mfcc)