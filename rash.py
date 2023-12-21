import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
import numpy as np

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.plot_data()

    def plot_data(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        self.ax.plot(x, y)
        self.ax.set_title('Matplotlib Plot')
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.central_widget = MatplotlibWidget(self)
        self.setCentralWidget(self.central_widget)

        self.button = QPushButton('Update Plot', self)
        self.button.clicked.connect(self.update_plot)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.central_widget)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def update_plot(self):
        self.central_widget.plot_data()

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
