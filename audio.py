from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal

AudioName = "e.wav" # Audio File
fs, Audiodata = wavfile.read(AudioName)

# spectrum
n = len(Audiodata)
AudioFreq = fft(Audiodata)
AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))]
MagFreq = np.abs(AudioFreq)
MagFreq = MagFreq / float(n)
MagFreq = MagFreq**2
if n % 2 > 0: # ffte odd
    MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
else:# fft even
    MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2

# spectrogram
N = 512 #Number of point in the fft
f, t, Sxx = signal.spectrogram(Audiodata, fs, window=signal.blackman(N), nfft=N)

# Plot the audio signal in time
plt.plot(Audiodata)
plt.title('Audio signal in time', size=16)

# Plot the power spectrum
freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n)
plt.figure()
plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq))
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power spectrum (dB)')

# Plot the spectrogram
plt.figure()
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [seg]')
plt.title('Spectrogram with scipy.signal', size=16)
plt.show()