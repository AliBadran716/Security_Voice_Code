from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUiType
import pyqtgraph as pg
import cv2
import numpy as np
import pandas as pd
import os
import sys
from os import path
import wave
import threading
import pyaudio
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,QFileDialog
from recording import AudioRecorder
from scipy.signal import spectrogram
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from voice import Voice
# from matplotlib import MatplotlibWidget


FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_button()
        self.passwords = {
            "unlock_the_gate": ["unlock_the_gate.wav"    #wav file
                                , None                   #similarity
                                , None],                 #instance
            "open_middle_door": ["open_middle_door.wav" 
                                 , None 
                                 , None   ],
            "grant_me_access": ["grant_me_access.wav" 
                                , None 
                                , None   ],
        }

    def handle_button(self):
        self.pushButton.clicked.connect(self.start_recording)

    def start_recording(self):
        print("Start Recording")
        if (self.pushButton.text() == "Start"):
            self.pushButton.setText("Stop")
            self.recorder = AudioRecorder()
            self.recorder.start_recording()
        else:
            self.pushButton.setText("Start")
            self.recorder.stop_recording()
            self.spectrogram(self.recorder.frames,44100,self.widget)
            self.spectrogram(self.recorder.frames,44100,self.widget)
            self.recorder.save_audio('trial.wav')
            test_voice = Voice('trial.wav')
            for password in self.passwords:
                self.passwords[password][2] = Voice(self.passwords[password][0])    
            for password in self.passwords:
                self.passwords[password][1] = self.check_similarity(test_voice, self.passwords[password][2])
            max = 0
            for password in self.passwords:
                if self.passwords[password][1] > max:
                    max = self.passwords[password][1]
                    max_password = password

            self.label.setText(max_password)



    
    def check_similarity(self, test_voice, password_voice):
        features1 = test_voice.get_amplitude()
        features2 = password_voice.get_amplitude()  
        features1, features2 = self.match_signal_length(features1, features2)
        
        # similarity_score = np.linalg.norm(features1 - features2)
        similarity_score = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return similarity_score
    
    def match_signal_length(self, signal1, signal2):
        max_length = max(len(signal1), len(signal2))
        signal1 = np.pad(signal1, (0, max_length - len(signal1)), 'constant')
        signal2 = np.pad(signal2, (0, max_length - len(signal2)), 'constant')
        return signal1, signal2
    
    def spectrogram(self, data, sampling_rate,widget):
        if widget.layout() is not None:
            print("Deleting layout")
            widget.layout().deleteLater()
        print("Spectrogram")
        data = np.frombuffer(b''.join(data), dtype=np.int16)
        print(data)
        _, _, Sxx = spectrogram(data, sampling_rate)
        time_axis = np.linspace(0, len(data) / sampling_rate, num=Sxx.shape[1])
        fig = Figure()
        fig = Figure(figsize=(2,3))
        ax = fig.add_subplot(111)
        ax.imshow(10 * np.log10(Sxx), aspect='auto', cmap='viridis',extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        widget.setLayout(layout)

def main():  # method to start app
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()