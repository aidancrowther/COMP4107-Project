# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
import imageio

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

#audio_path = librosa.util.example_audio_file()

# or uncomment the line below and point it at your favorite song:
#
audio_path = 'Data/Songs/107535.mp3'

sr = 8000
numMels = 32

y, sr = librosa.load(audio_path, sr=sr)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=numMels)

print(S.shape)
print(S[0][0])

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)
#log_S = normalize(log_S)

max = 0.0
min = 0.0

for i in log_S:
    for j in i:
        if(j < min): min = j
        if(j > max): max = j

print(min)
print(max)

#imageio.imwrite('test.png', log_S)

print(log_S.shape)
print(log_S[0][0])

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.show()