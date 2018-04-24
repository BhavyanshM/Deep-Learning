import pyaudio
from pylab import *
import wave
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import numpy as np
import peakutils.peak
import subprocess


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
    thres=50000/max(p))
frs = locs/5
frs = frs[frs>900]
upper = frs[frs>1300]
lower = frs[frs<1100]
mid = frs[(frs>1100)&(frs<1300)]
print(lower)
print(mid)
print(upper)
middle = np.mean(mid)
down = np.mean(lower)
up = np.mean(upper)
# plt.xlim(0, 10000)
# plt.plot(p,'b') 
# plt.axvline(x=down*5, color='k', linestyle='--')
# plt.axvline(x=middle*5, color='k', linestyle='--')
# plt.axvline(x=up*5, color='k', linestyle='--')
# plt.axhline(y=50000, color='y', linestyle='--')
# plt.show()

print(down)
print(middle)
print(up)

if (down > 995) and (down < 1005) and (middle>1195) and (middle<1205):
	if (up > 1395) and (up < 1405):
		print('Access Granted!')
		# with file('treasure.txt') as f:
		# 	string = f.read()
		# 	for c in string:
		# 		print(c)
		p = subprocess.Popen('notepad treasure.txt')
else :
	print('Could Not Authenticate!')