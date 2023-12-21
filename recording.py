import pyaudio
import wave

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []

    def start_recording(self):
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024,
                                      stream_callback=self.callback)
        self.frames = []

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def save_audio(self, file_name):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

# Example usage:
recorder = AudioRecorder()
recorder.start_recording()

# ... wait for some time or until a stop button is pressed ...

recorder.stop_recording()
recorder.save_audio('output.wav')
