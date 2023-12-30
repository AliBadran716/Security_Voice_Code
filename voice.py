import scipy.io.wavfile
import librosa
import numpy as np

class Voice:
    def __init__(self , voice_path):
        sampling_rate , self.voice = scipy.io.wavfile.read(voice_path) 
        self.sampling_rate = sampling_rate
        self.mv , self.msr = librosa.load(voice_path)
        self.compute_fourier_transform()
        
    
    def compute_fourier_transform(self):
        self.fourier_transform = np.fft.fft(self.voice)
        self.frequencies = np.fft.fftfreq(len(self.voice) , 1/self.sampling_rate)
        self.frequencies = self.frequencies[:len(self.frequencies)//2]
        self.fourier_transform = self.fourier_transform[:len(self.fourier_transform)//2]
    
    
        
    def get_amplitude(self):
        return np.abs(self.fourier_transform)
    
    def get_stft(self):
        return np.abs(librosa.stft(self.mv))    

    def get_central_centroid(self):
        return np.sum(np.abs(self.frequencies) * self.get_amplitude()) / np.sum(self.get_amplitude())
    
    def get_phase(self):
        return np.angle(self.fourier_transform)
    
    def extract_features(self):

        self.mv = np.array(self.mv)
        self.mv = self.mv.astype(float)


        mfccs = librosa.feature.mfcc(y=self.mv, sr=self.msr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        return np.vstack([mfccs, delta_mfccs, delta2_mfccs])
