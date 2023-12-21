import scipy.io.wavfile
import numpy as np

class Voice:
    def __init__(self , voice_path):
        sampling_rate , self.voice = scipy.io.wavfile.read(voice_path) 
        self.sampling_rate = 44100
        self.compute_fourier_transform()
        
    
    def compute_fourier_transform(self):
        self.fourier_transform = np.fft.fft(self.voice)
        self.frequencies = np.fft.fftfreq(len(self.voice) , 1/self.sampling_rate)
        self.frequencies = self.frequencies[:len(self.frequencies)//2]
        self.fourier_transform = self.fourier_transform[:len(self.fourier_transform)//2]
    
    
        
    def get_amplitude(self):
        return np.abs(self.fourier_transform)
    
    def get_central_centroid(self):
        return np.sum(np.abs(self.frequencies) * self.get_amplitude()) / np.sum(self.get_amplitude())
    
    def get_phase(self):
        return np.angle(self.fourier_transform)
