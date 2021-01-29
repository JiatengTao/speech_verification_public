from scipy.io import wavfile
import webrtcvad
import struct
from scipy.io.wavfile import write
import os
import numpy as np
# matplotlib inline
import matplotlib.pyplot as plt

train_audio_path = "C:/Users/10856/Desktop/MP3 to WAV/"
filename = "output.wav"
sample_rate, samples = wavfile.read(os.path.join(train_audio_path, filename))
vad = webrtcvad.Vad()
vad.set_mode(3)

raw_samples = struct.pack("%dh" % len(samples), *samples)

window_duration = 0.03 
samples_per_window = int(window_duration * sample_rate + 0.4)
bytes_per_sample = 2

for start in np.arange(0, len(samples), samples_per_window):
    stop = min(start + samples_per_window, len(samples))
    is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], sample_rate = sample_rate)
    segments.append(dict(start = start,stop = stop,is_speech = is_speech))

plt.figure(figsize = (10,7))
plt.plot(samples)

ymax = max(samples)

# plot segment identifed as speech
for segment in segments:
    if segment['is_speech']:
        plt.plot([ segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color = 'orange')

plt.xlabel('sample')
plt.grid()

speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
new_path = "C:/Users/10856/Desktop/MP3 to WAV/cleaned.wav"
write(new_path, sample_rate, speech_samples)