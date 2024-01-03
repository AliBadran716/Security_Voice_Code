import sys
from os import path

import functools
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import convolve2d

from recording import AudioRecorder
from voice import Voice
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PyQt5.QtCore import QTimer
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
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
        self.passwords = {
            "unlock_the_gate": [
                "unlock_the_gate.wav"
                 
                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 20  # acceptable similarity factor
            ],
            "open_middle_door": [
                "open_middle_door.wav"
                 
                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 12  # acceptable similarity factor
            ],
            "grant_me_access": [
                "grant_me_access.wav"
                 
                 
                 
                # list of wav files stored in voice_data_base folder
                , None  # similarity factor
                , None  # instance
                , True  # Have access
                , 11  # acceptable similarity factor
            ],
        }
        # voice
        self.voice = {
            "ahmed_ali": [
                ["voice_data_base/ahmed_ali_unlock_the_gate_0.wav", "voice_data_base/ahmed_ali_unlock_the_gate_1.wav",
                 "voice_data_base/ahmed_ali_unlock_the_gate_2.wav", "voice_data_base/ahmed_ali_unlock_the_gate_3.wav",
                 "voice_data_base/ahmed_ali_open_middle_door_0.wav", "voice_data_base/ahmed_ali_open_middle_door_1.wav",
                 "voice_data_base/ahmed_ali_open_middle_door_2.wav", "voice_data_base/ahmed_ali_open_middle_door_3.wav",
                 "voice_data_base/ahmed_ali_grant_me_access_0.wav", "voice_data_base/ahmed_ali_grant_me_access_1.wav",
                 "voice_data_base/ahmed_ali_grant_me_access_2.wav", "voice_data_base/ahmed_ali_grant_me_access_3.wav"],
                None,  # similarity factor
                [None, None, None, None, None, None, None, None, None, None, None, None],
                True,  # Have access
                50  # acceptable similarity factor
            ],
            "bedro": [["voice_data_base/bedro_unlock_the_gate_0.wav", "voice_data_base/bedro_unlock_the_gate_1.wav",
                       "voice_data_base/bedro_unlock_the_gate_2.wav", "voice_data_base/bedro_unlock_the_gate_3.wav",
                       "voice_data_base/bedro_open_middle_door_0.wav", "voice_data_base/bedro_open_middle_door_1.wav",
                       "voice_data_base/bedro_open_middle_door_2.wav", "voice_data_base/bedro_open_middle_door_3.wav",
                       "voice_data_base/bedro_grant_me_access_0.wav", "voice_data_base/bedro_grant_me_access_1.wav",
                       "voice_data_base/bedro_grant_me_access_2.wav", "voice_data_base/bedro_grant_me_access_3.wav"],
                      None,  # similarity factor
                      [None, None, None, None, None, None, None, None, None, None, None, None],
                      True,  # Have access
                      80  # acceptable similarity factor
                      ],

            "muhannad":[["voice_data_base/muhannad_unlock_the_gate_0.wav" , "voice_data_base/muhannad_unlock_the_gate_1.wav",
                            "voice_data_base/muhannad_unlock_the_gate_2.wav" , "voice_data_base/muhannad_unlock_the_gate_3.wav",
                            "voice_data_base/muhannad_open_middle_door_0.wav" , "voice_data_base/muhannad_open_middle_door_1.wav",
                            "voice_data_base/muhannad_open_middle_door_2.wav" ,
                            "voice_data_base/muhannad_open_middle_door_3.wav",
                            "voice_data_base/muhannad_grant_me_access_0.wav", "voice_data_base/muhannad_grant_me_access_1.wav",
                            "voice_data_base/muhannad_grant_me_access_2.wav", "voice_data_base/muhannad_grant_me_access_3.wav"],

            None, #similarity factor
            [None, None, None, None, None, None, None, None, None, None, None, None],
            True
            ,50
            ],

            "hassan": [["voice_data_base/hassan_unlock_the_gate_0.wav", "voice_data_base/hassan_unlock_the_gate_1.wav",
                        "voice_data_base/hassan_unlock_the_gate_2.wav", "voice_data_base/hassan_unlock_the_gate_3.wav",
                        "voice_data_base/hassan_open_middle_door_0.wav", "voice_data_base/hassan_open_middle_door_1.wav",
                        "voice_data_base/hassan_open_middle_door_2.wav",
                        "voice_data_base/hassan_open_middle_door_3.wav",
                        "voice_data_base/hassan_grant_me_access_0.wav", "voice_data_base/hassan_grant_me_access_1.wav",
                        "voice_data_base/hassan_grant_me_access_2.wav", "voice_data_base/hassan_grant_me_access_3.wav"],
                       None,  # similarity factor
                       [None, None, None, None, None, None, None, None, None, None, None, None],
                       True,  # Have access
                       50  # acceptable similarity factor
                       ]
        }

        self.create_password_audio_instance()  # create instance of each password
    
    
    

    def check_word_similarity(self , tested , passwrd , passw):
        tested_data , sampling_rate = tested.get_voice()
        passwrd_data , sampling_rate = passwrd.get_voice()
        normalized_tested_data = tested_data / np.max(np.abs(tested_data))
        normalized_passwrd_data = passwrd_data / np.max(np.abs(passwrd_data))
        test_mfcc,_ = tested.extract_features_new(sampling_rate , normalized_tested_data)
        pass_mfcc,_ = passwrd.extract_features_new(sampling_rate , normalized_passwrd_data)
        distance, _  = fastdtw(test_mfcc, pass_mfcc, dist=euclidean)
        print(f"{passw} : {distance}")
        return distance
    
    
    
    
    
    
    def handle_button(self):
        self.record_btn.clicked.connect(self.start_recording)
        # connect combox box word_access_i from 1 to 3 to the function change_word_access
        for i in range(1, 4):
            getattr(self, 'word_access_' + str(i)).clicked.connect(self.change_word_access)

        # connect combox box person_access_i from 1 to 4 to the function change_voice_access
        for i in range(1, 5):
            getattr(self, 'person_access_' + str(i)).clicked.connect(self.change_word_access)

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

        for voice in (self.voice.keys()):
            for i in range(len(self.voice[voice][0])):
                self.voice[voice][2][i] = Voice(self.voice[voice][0][i])

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
            for voice in self.voice:
                self.voice[voice][1] = self.check_similarity(test_voice, self.voice[voice][2], voice, False)
            for password in self.passwords:
                self.passwords[password][1] = self.check_word_similarity(test_voice, self.passwords[password][2], password)
            max_similarity = 0
            max_voice = 0
            Access = ""

            for i, password in enumerate(self.passwords):
                # set word_perc_i and word_bar_i from 1 to 3 to the similarity factor percentage to two decimal places
                factor_percentage = round(self.passwords[password][1], 2)
                getattr(self, 'word_perc_' + str(i + 1)).setText(str(factor_percentage))
                getattr(self, 'word_bar_' + str(i + 1)).setValue(int(factor_percentage))

                if self.passwords[password][1] > max_similarity:
                    max_similarity = self.passwords[password][1]
                    similar_word = password
                    if max_similarity < self.passwords[similar_word][4] or not self.passwords[similar_word][3]:
                        Access = "Access Denied"
                    else:
                        Access = "Access Granted"

            for i, voice in enumerate(self.voice):
                # set person_perc_i and person_bar_i from 1 to 4 to the similarity factor percentage
                factor_percentage = round(self.voice[voice][1], 2)
                getattr(self, 'person_perc_' + str(i + 1)).setText(str(factor_percentage))
                getattr(self, 'person_bar_' + str(i + 1)).setValue(int(factor_percentage))

                if self.voice[voice][1] > max_voice:
                    max_voice = self.voice[voice][1]
                    similar_person = voice
                    if max_voice <  self.voice[similar_person][4] or not self.voice[similar_person][3]:
                        Access = "Access Denied"
                    else:
                        Access = "Access Granted"

            self.label.setText(similar_word)
            self.label_2.setText(similar_person)
            self.access_label.setText(Access)


    def check_similarity(self, test_voice, password_voice, keyword, word=True):
        similarity_score = []
        correlation = []
        convulotion = []

        for i in range(len(password_voice)):
            if word:
                features1 = test_voice.get_stft()
                features2 = password_voice[i].get_stft()
            else:
                features1 = test_voice.extract_features()
                features2 = password_voice[i].extract_features()

            features1, features2 = self.match_signal_length(features1, features2)

            # similarity_score = np.linalg.norm(features1 - features2)
            similarity_score.append(
                np.mean(np.dot(features1.T, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))))
            # Compute Pearson correlation coefficient
            correlation.append(np.corrcoef(features1.flatten(), features2.flatten())[0, 1])
            # Compute convolution
            convulotion.append(convolve2d(features1, features2, mode='valid'))

        similarity_score = np.mean(similarity_score) * 10000
        correlation = np.mean(correlation) * 100
        convulotion = np.mean(convulotion)

        # print(keyword)
        # print('correlation', correlation)
        # print("similarity_score", similarity_score)
        # print("convulotion", convulotion)
        # print("")

        return correlation

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