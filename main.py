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

        self.detected_word = ""
        self.detected_person = ""

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
                ,
                ["voices/unlock_the_gate/ahmed_ali_unlock_the_gate.wav", "voices/unlock_the_gate/bedro_unlock_the_gate.wav",
                 "voices/unlock_the_gate/hassan_unlock_the_gate.wav",
                 "voices/unlock_the_gate/muhannad_unlock_the_gate.wav"],
                ["ahmed_ali", "ali_badran", "hassan", "muhannad"],
                [None, None, None, None]
            ],
            "open_middle_door": [
                "open_middle_door.wav"

                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 16000  # acceptable similarity factor
                ,
                ["voices/open_middle_door/ahmed_ali_open_middle_door.wav", "voices/open_middle_door/bedro_open_middle_door.wav",
                 "voices/open_middle_door/hassan_open_middle_door.wav",
                 "voices/open_middle_door/muhannad_open_middle_door.wav"],
                ["ahmed_ali", "ali_badran", "hassan", "muhannad"],
                [None, None, None, None]

            ],
            "grant_me_access": [
                "grant_me_access.wav"

                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 16000  # acceptable similarity factor
                ,
                ["voices/grant_me_access/ahmed_ali_grant_me_access.wav", "voices/grant_me_access/bedro_grant_me_access.wav",
                 "voices/grant_me_access/hassan_grant_me_access.wav",
                 "voices/grant_me_access/muhannad_grant_me_access.wav"],
                ["ahmed_ali", "ali_badran", "hassan", "muhannad"],
                [None, None, None, None]

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
            "ali_badran": [[],
                      None,  # similarity factor
                      [None],
                      True,  # Have access
                      10  # acceptable similarity factor
                      ],

            "muhannad": [[],
                         None,  # similarity factor
                         [None],
                         True,
                         10
                         ],

            "hassan": [[],
                       None,  # similarity factor
                       [None],
                       True,  # Have access
                       10  # acceptable similarity factor
                       ]
        }

        self.create_password_audio_instance()  # create instance of each password

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





            for password in self.passwords:
                self.passwords[password][1] = self.check_word_similarity(test_voice, self.passwords[password][2],
                                                                         password)

            # make the distance as a percentage where the lowest distance is the highest percentage but not 100 %

            min_similarity = 16000  # Adjusted the minimum similarity threshold

            Access = ""
            within_range = False

            # Iterate through the passwords and compare distances
            for i, password in enumerate(self.passwords):
                # Check if the minimum distance is below the acceptable threshold
                if self.passwords[password][1] < min_similarity and self.passwords[password][3]:
                    within_range = True
                    # Calculate the percentage based on the smaller of the two distances
                    factor_percentage = round((12000 / self.passwords[password][1]) * 100, 3)
                else:
                    factor_percentage = round((10000 / self.passwords[password][1]) * 100, 3)

                # Update the UI elements (assuming you are working with a GUI)
                getattr(self, 'word_perc_' + str(i + 1)).setText(str(factor_percentage))
                getattr(self, 'word_bar_' + str(i + 1)).setValue(int(factor_percentage))

            # Get the detected word based on the minimum distance
            self.detected_word = min(self.passwords, key=lambda x: self.passwords[x][1])
            self.check_person_similarity(test_voice)

            min_similarity = 16000  # Adjusted the minimum similarity threshold

            within_range_person = False
            min_similarity = 16000  # Adjusted the minimum similarity threshold
            # Iterate through the voices and compare distances
            for i, voice in enumerate(self.voice):
                # Check if the minimum distance is below the acceptable threshold
                if self.voice[voice][1] < min_similarity and self.voice[voice][3]:
                    within_range_person = True
                    # Calculate the percentage based on the smaller of the two distances
                    factor_percentage = round((12000 / self.voice[voice][1]) * 100, 3)
                else:
                    factor_percentage = round((10000 / self.voice[voice][1]) * 100, 3)

                # Update the UI elements (assuming you are working with a GUI)
                getattr(self, 'person_perc_' + str(i + 1)).setText(str(factor_percentage))
                getattr(self, 'person_bar_' + str(i + 1)).setValue(int(factor_percentage))

            self.detected_person = min(self.voice, key=lambda x: self.voice[x][1])


            # Determine access based on whether any password is within the range
            if within_range and within_range_person:
                Access = "Access Granted"
            else:
                Access = "Access Denied"

            # Update the UI based on the prediction
            self.label_2.setText(f"Prediction: {predicted_class}")
            self.label.setText(self.detected_word)
            self.access_label.setText(Access)

    def check_person_similarity(self, tested):
        tested_data, sampling_rate = tested.get_voice()
        normalized_tested_data = tested_data / np.max(np.abs(tested_data))
        tested_mfcc, _ = tested.extract_features_new(sampling_rate, normalized_tested_data)
        tested_features = tested_mfcc

        for i in range(len(self.passwords[self.detected_word][5])):
            voice_instance = Voice(self.passwords[self.detected_word][5][i])
            person_data, sampling_rate = voice_instance.get_voice()
            normalized_person_data = person_data / np.max(np.abs(person_data))
            person_mfcc, _ = voice_instance.extract_features_new(sampling_rate, normalized_person_data)
            self.passwords[self.detected_word][7][i] = person_mfcc

        for i, key in enumerate(self.passwords[self.detected_word][6]):
            dist, _ = fastdtw(tested_features, self.passwords[self.detected_word][7][i], dist=euclidean)
            self.voice[key][1]= dist
            # print(f"{self.voice[key]} : {self.voice[key][1]}")


    def check_word_similarity(self, tested, passwrd, passw):
        tested_data, sampling_rate = tested.get_voice()
        passwrd_data, sampling_rate = passwrd.get_voice()

        normalized_tested_data = tested_data / np.max(np.abs(tested_data))
        normalized_passwrd_data = passwrd_data / np.max(np.abs(passwrd_data))

        test_mfcc, _ = tested.extract_features_new(sampling_rate, normalized_tested_data)
        pass_mfcc, _ = passwrd.extract_features_new(sampling_rate, normalized_passwrd_data)
        distance, _ = fastdtw(test_mfcc, pass_mfcc, dist=euclidean)

        # print(f"{passw} : {distance}")
        self.passwords[passw][1] = distance
        return distance


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
