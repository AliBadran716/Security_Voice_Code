import scipy.io.wavfile
import librosa
import numpy as np

class Voice:
    def __init__(self , voice_path):
        sampling_rate , self.voice = scipy.io.wavfile.read(voice_path) 
        self.sampling_rate = sampling_rate
        self.mv , self.msr = librosa.load(voice_path)


    def get_stft(self):
        return np.abs(librosa.stft(self.mv))    

    def extract_features(self):

        self.mv = np.array(self.mv)
        self.mv = self.mv.astype(float)


        mfccs = librosa.feature.mfcc(y=self.mv, sr=self.msr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        return np.vstack([mfccs, delta_mfccs, delta2_mfccs])
