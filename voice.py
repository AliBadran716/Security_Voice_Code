import scipy.io.wavfile
import librosa
import numpy as np


class Voice:
    def __init__(self, voice_path):
        sampling_rate, self.voice = scipy.io.wavfile.read(voice_path)
        self.sampling_rate = sampling_rate
        self.mv, self.msr = librosa.load(voice_path)

    def extract_features_new(self, s_rate, data):
        mv = np.array(data)
        mv = mv.astype(float)
        mfccs = librosa.feature.mfcc(y=mv, sr=s_rate, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfccs_1 = mfccs.T
        return mfccs_1, np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    def get_voice(self):
        return self.mv, self.msr
