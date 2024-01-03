import sys
from os import path
import statistics
import functools
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import convolve2d
from os import path
from scipy.stats import mode
import pickle
from recording import AudioRecorder
from voice import Voice
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PyQt5.QtCore import QTimer
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import librosa
import pandas as pd

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_button()
        self.recorder = AudioRecorder()
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.start_recording)
        with open('model.pkl', 'rb') as model_file:
            self.rf_model = pickle.load(model_file)

        # Load the scaler
        with open('scaler.pkl', 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
        self.passwords = {
            "unlock_the_gate": [
                "unlock_the_gate.wav"

                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 16000  # acceptable similarity factor
            ],
            "open_middle_door": [
                "open_middle_door.wav"

                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 16000  # acceptable similarity factor
            ],
            "grant_me_access": [
                "grant_me_access.wav"

                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 16000  # acceptable similarity factor
            ],
        }
        # voice
        self.voice = {
            "ahmed_ali": [
                [],
                None,  # similarity factor
                [None],
                True,  # Have access
                10  # acceptable similarity factor
            ],
            "bedro": [[],
                      None,  # similarity factor
                      [None],
                      True,  # Have access
                      10  # acceptable similarity factor
                      ],

            "muhannad": [[],

                         None,  # similarity factor
                         [None],
                         True
                , 10
                         ],

            "hassan": [[],
                       None,  # similarity factor
                       [None],
                       True,  # Have access
                       10  # acceptable similarity factor
                       ]
        }

        self.create_password_audio_instance()  # create instance of each password

    def check_word_similarity(self, tested, passwrd, passw):
        tested_data, sampling_rate = tested.get_voice()
        passwrd_data, sampling_rate = passwrd.get_voice()

        normalized_tested_data = tested_data / np.max(np.abs(tested_data))
        normalized_passwrd_data = passwrd_data / np.max(np.abs(passwrd_data))

        test_mfcc, _ = tested.extract_features_new(sampling_rate, normalized_tested_data)
        pass_mfcc, _ = passwrd.extract_features_new(sampling_rate, normalized_passwrd_data)
        distance, _ = fastdtw(test_mfcc, pass_mfcc, dist=euclidean)

        print(f"{passw} : {distance}")
        self.passwords[passw][1] = distance
        return distance

    def handle_button(self):
        self.record_btn.clicked.connect(self.start_recording)
        # connect combox box word_access_i from 1 to 3 to the function change_word_access
        for i in range(1, 4):
            getattr(self, 'word_access_' + str(i)).clicked.connect(self.change_word_access)

        # connect combox box person_access_i from 1 to 4 to the function change_voice_access
        for i in range(1, 5):
            getattr(self, 'person_access_' + str(i)).clicked.connect(self.change_word_access)

    def extract_mfccs(self, file_path):
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs = mfccs.T
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs = delta_mfccs

        # (216) rows = frames ,  and (26) columns = features
        features = np.hstack([mfccs, delta_mfccs])
        feature_names = [f"MFCC_{i + 1}" for i in range(features.shape[1])]
        return pd.DataFrame(features, columns=feature_names)

    def extract_pitch(self, file_path):
        y, sr = librosa.load(file_path)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.85])
        return pd.DataFrame({"Pitch": [pitch]})

    def extract_chroma(self, file_path):
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = chroma.T
        feature_names = [f"Chroma_{i + 1}" for i in range(chroma.shape[1])]
        return pd.DataFrame(chroma, columns=feature_names)

    def extract_zero_crossings(self, file_path):
        y, sr = librosa.load(file_path)
        zero_crossings = librosa.feature.zero_crossing_rate(y)
        zero_crossings = zero_crossings.T
        feature_names = [f"ZeroCrossings_{i + 1}" for i in range(zero_crossings.shape[1])]
        return pd.DataFrame(zero_crossings, columns=feature_names)

    def extract_spectral_contrast(self, file_path):
        y, sr = librosa.load(file_path)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast = spectral_contrast.T
        feature_names = [f"SpectralContrast_{i + 1}" for i in range(spectral_contrast.shape[1])]
        return pd.DataFrame(spectral_contrast, columns=feature_names)

    def change_word_access(self):
        # change the access of the word
        for i, password in enumerate(self.passwords):
            is_checked = getattr(self, 'word_access_' + str(i + 1)).isChecked()
            self.passwords[password][3] = is_checked

        for i, voice in enumerate(self.voice):
            is_checked = getattr(self, 'person_access_' + str(i + 1)).isChecked()
            self.voice[voice][3] = is_checked

    def create_password_audio_instance(self):
        for password in (self.passwords.keys()):
            # for i in range(len(self.passwords[password][0])):
            self.passwords[password][2] = Voice(self.passwords[password][0])

    def start_recording(self):
        if self.record_btn.text() == "Start":
            # print("Start Recording")
            self.record_btn.setText("Stop")
            self.recorder = AudioRecorder()
            self.recorder.start_recording()
            self.recording_timer.start(2500)  # Set timer for 2 seconds
        else:
            # print("Stop Recording")
            self.record_btn.setText("Start")
            self.recording_timer.stop()  # Stop the timer
            self.recorder.stop_recording()
            self.spectrogram(self.recorder.frames, 44100, self.widget)
            self.recorder.save_audio('trial.wav')
            test_voice = Voice('trial.wav')
            # Extract features from the recorded voice using the provided functions
            test_mfccs = self.extract_mfccs('trial.wav')
            test_chroma = self.extract_chroma('trial.wav')
            test_zero_crossings = self.extract_zero_crossings('trial.wav')
            test_spectral_contrast = self.extract_spectral_contrast('trial.wav')
            test_features = pd.concat([test_mfccs, test_chroma, test_zero_crossings, test_spectral_contrast], axis=1)
            # Scale the features using the loaded scaler
            test_features_scaled = self.scaler.transform(test_features)

            # Make predictions using the loaded Random Forest model
            predictions = self.rf_model.predict(test_features_scaled)
            # Assuming predictions is your array of predictions
            predicted_class = np.unique(predictions, return_counts=True)
            predicted_class = predicted_class[0][np.argmax(predicted_class[1])]
            # predicted_class = predictions[0]
            print(predicted_class)
            # Update the UI based on the prediction
            self.label_2.setText(f"Prediction: {predicted_class}")

            for password in self.passwords:
                self.passwords[password][1] = self.check_word_similarity(test_voice, self.passwords[password][2],
                                                                         password)

            # make the distance as a percentage where the lowest distance is the highest percentage but not 100 %


            max_similarity = 0
            Access = ""

            # if the minimum distance is greater than the acceptable distance which is 16000 then the user has no access
            # else the user has access and set the access to the password that has the minimum distance and the progress bar to the percentage of the distance
            for i, password in enumerate(self.passwords):
                if self.passwords[password][1] < self.passwords[password][4]:
                    Access = 'Access Granted'
                    max_similarity = self.passwords[password][1]
                    factor_percentage = int((1 - (self.passwords[password][1] / self.passwords[password][4])) * 100)
                    getattr(self, 'word_perc_' + str(i + 1)).setText(str(factor_percentage))
                    getattr(self, 'word_bar_' + str(i + 1)).setValue(int(factor_percentage))

                    break
                else:
                    Access = "Access Denied"
                    max_similarity = self.passwords[password][1]
                    self.progressBar.setValue(0)

            self.label.setText(max_similarity)
            self.access_label.setText(Access)

    def match_signal_length(self, signal1, signal2):
        len1, len2 = signal1.shape[1], signal2.shape[1]
        max_len = max(len1, len2)

        if len1 < max_len:
            signal1 = np.pad(signal1, ((0, 0), (0, max_len - len1)), 'constant')
        elif len2 < max_len:
            signal2 = np.pad(signal2, ((0, 0), (0, max_len - len2)), 'constant')

        return signal1, signal2

    def spectrogram(self, data, sampling_rate, widget):
        if widget.layout() is not None:
            widget.layout().deleteLater()

        data = np.frombuffer(b''.join(data), dtype=np.int16)
        # print(data)
        fig = Figure()
        fig = Figure(figsize=(2, 3))
        ax = fig.add_subplot(111)
        Sxx = ax.specgram(data, Fs=sampling_rate, cmap='plasma')
        # time_axis = np.linspace(0, len(data) / sampling_rate, num=Sxx.shape[1])

        # ax.imshow(10 * np.log10(Sxx), aspect='auto', cmap='viridis',extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        widget.setLayout(layout)
        widget.show()


def main():  # method to start app
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()
