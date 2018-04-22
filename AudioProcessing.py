import pyaudio
from pylab import *
import wave
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import numpy as np
import peakutils.peak

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECS = 5
WAVE_OUTPUT_NAME = "experiment.wav"

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("recording")

frames = []

for i in range(0, int(RATE/CHUNK * RECORD_SECS)):
	data = stream.read(CHUNK)
	frames.append(data)

print("finised recording")

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_NAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

fs, data = wf.read(WAVE_OUTPUT_NAME)
a = data.T[0] # this is a two channel soundtrack, I get the first track
b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = int(len(c)/2)  # you only need half of the fft list (real signal symmetry)`
p = abs(c[:(d-1)])
locs = peakutils.peak.indexes(np.array(p),
    thres=20000/max(p))
print(np.mean(locs))
plt.xlim(0, 10000)
plt.plot(p,'b') 
plt.show()


