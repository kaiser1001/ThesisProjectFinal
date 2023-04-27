import os
import numpy as np
import scipy.io.wavfile as wav
from mpmath import eps
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import warnings
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")

# Define the number of MFCC features and HMM states
rate = 16000
num_features = 13
num_states = 13
train_pct = 0.8
# Define the directory containing the training data
data_dir = "thesisSpeechData"
# Define the names of the speaker subdirectories in the training data directory
speakers = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]
# Define window size and shift for frame blocking
win_size = 0.025
win_shift = 0.01
# Define number of filterbanks
num_filterbanks = 26
# Define pre-emphasis coefficient
pre_emphasis = 0.97
# Define the lower and upper frequency bounds of the filterbank
lower_freq = 0
upper_freq = 8000

# Create a list of Mel filterbank frequencies
mel_freqs = np.linspace(lower_freq, upper_freq, num_filterbanks + 2)
hz_freqs = mel_freqs
bin_width = np.floor((win_size * rate) / 2)
bin_idx = np.floor(hz_freqs / (rate / (2 * bin_width)))

# Create a list to hold the training feature vectors and labels
X_train = []
y_train = []
np.set_printoptions(threshold=np.inf)
#Training
# Loop over the speaker subdirectories
for speaker in speakers:
    # Get the path to the speaker's training subdirectory
    train_dir = os.path.join(data_dir, speaker)
    # Loop over the audio files in the speaker's training subdirectory
    for file in os.listdir(train_dir):
        # Get the path to the audio file
        file_path = os.path.join(train_dir, file)
        # Load the audio signal
        rate, signal = wav.read(file_path)
        # Apply pre-emphasis
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        # Apply Hamming window
        window = np.hamming(int(win_size*rate))
        num_frames = int(np.ceil(len(signal)/rate/win_shift))
        pad_len = num_frames*int(win_shift*rate) + int(win_size*rate) - len(signal)
        signal = np.pad(signal, (0, pad_len), mode='constant')
        frames = np.zeros((num_frames, int(win_size*rate)))
        for i in range(num_frames):
            frames[i] = signal[int(i*win_shift*rate):int(i*win_shift*rate+win_size*rate)]
            frames[i] *= window
        # Compute power spectrum
        power_spectrum = np.square(np.abs(np.fft.rfft(frames, n=int(win_size * rate)-1)))
        # Compute Mel filterbank
        filterbank = np.zeros((num_filterbanks, int(bin_width)))
        for i in range(num_filterbanks):
            l_bin = int(bin_idx[i])
            m_bin = int(bin_idx[i + 1])
            r_bin = int(bin_idx[i + 2])
            for j in range(m_bin - l_bin):
                filterbank[i, j + l_bin] = (j + 1) / (mel_freqs[i + 1] - mel_freqs[i] + eps)
            for j in range(r_bin - m_bin):
                filterbank[i, r_bin - j - 1] = (j + 1) / (mel_freqs[i + 2] - mel_freqs[i + 1] + eps)
        # Compute Mel filterbank energies
        mel_energies = np.dot(power_spectrum, filterbank.T)
        # Take the logarithm of the Mel filterbank energies
        log_mel_energies = np.log(mel_energies)
        # Compute MFCCs
        features = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :num_features]
        # Append the MFCCs and label to the training lists
        X_train.append(features)
        y_train.append(speaker)

# Convert the training data to a list of equal-sized numpy arrays
max_len = max(len(x) for x in X_train)
X_train = [np.pad(x, [(0, max_len - len(x)),(0, 0)], mode='constant') for x in X_train]

# Convert the training data to numpy arrays
X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train)

print("Rate: ",rate)
print("Signal: ",signal)
print("Frames: ",frames)
print("Hamming Window: ",window)
print("Mel Filterbank: ",filterbank)
print("Features: ",features)

# Create an HMM for each speaker in speaker identification
models_speaker = []
for speaker in speakers:
    # Get the training feature vectors and labels for the current speaker
    X = X_train[y_train == speaker]
    X = X.reshape(-1, X.shape[-1])
    # Standardize the features
    scaler = StandardScaler().fit(X)
    X = X.reshape(-1, X.shape[-1])
    # Create an HMM for the current speaker with L2 regularization
    model = hmm.GaussianHMM(n_components=num_states, min_covar=1e-3, algorithm="any")
    model.fit(X)
    # Add the HMM to the list of models
    models_speaker.append(model)

# Define the directory containing the test data
test_dir = os.path.join(data_dir, speaker)
conf_matrix = np.zeros((len(speakers), len(speakers)), dtype=int)

#Testing Speaker Identification
# Loop over the speaker subdirectories
for speaker_idx, speaker in enumerate(speakers):
    # Get the path to the speaker's test subdirectory
    test_dir = os.path.join(data_dir, speaker)
    # Shuffle the files in the test subdirectory and take a random subset
    test_files = os.listdir(test_dir)
    np.random.shuffle(test_files)
    test_files = test_files[:150]  # take a random subset of 150 files
    # Create variables to hold the test results
    correct = 0
    total = 0
    # Loop over the audio files in the speaker's test subdirectory
    for file in test_files:
        # Get the path to the audio file
        file_path = os.path.join(test_dir, file)
        # Load the audio signal
        rate, signal = wav.read(file_path)
        # Apply pre-emphasis
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        # Apply Hamming window
        window = np.hamming(int(win_size * rate))
        num_frames = int(np.ceil(len(signal) / rate / win_shift))
        pad_len = num_frames * int(win_shift * rate) + int(win_size * rate) - len(signal)
        signal = np.pad(signal, (0, pad_len), mode='constant')
        frames = np.zeros((num_frames, int(win_size * rate)))
        for i in range(num_frames):
            frames[i] = signal[int(i * win_shift * rate):int(i * win_shift * rate + win_size * rate)]
            frames[i] *= window
        # Compute power spectrum
        power_spectrum = np.square(np.abs(np.fft.rfft(frames, n=int(win_size * rate) - 1)))
        # Compute Mel filterbank
        filterbank = np.zeros((num_filterbanks, int(bin_width)))
        for i in range(num_filterbanks):
            l_bin = int(bin_idx[i])
            m_bin = int(bin_idx[i + 1])
            r_bin = int(bin_idx[i + 2])
            for j in range(m_bin - l_bin):
                filterbank[i, j + l_bin] = (j + 1) / (mel_freqs[i + 1] - mel_freqs[i] + eps)
            for j in range(r_bin - m_bin):
                filterbank[i, r_bin - j - 1] = (j + 1) / (mel_freqs[i + 2] - mel_freqs[i + 1] + eps)
        # Compute Mel filterbank energies
        mel_energies = np.dot(power_spectrum, filterbank.T)
        # Take the logarithm of the Mel filterbank energies
        log_mel_energies = np.log(mel_energies)
        # Compute MFCCs
        features = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :num_features]
        # Calculate the log-likelihood of the features for each speaker model
        likelihoods = [model.score(features) for model in models_speaker]
        # Get the index of the speaker with the highest log-likelihood
        prediction_idx = np.argmax(likelihoods)
        # Update the confusion matrix
        actual_idx = speaker_idx
        conf_matrix[actual_idx, prediction_idx] += 1
        # Update the test results
        if prediction_idx == actual_idx:
            correct += 1
        total += 1

    # Calculate the precision, recall, and accuracy for the speaker
    precision = conf_matrix[speaker_idx, speaker_idx] / np.sum(conf_matrix[:, speaker_idx])
    recall = conf_matrix[speaker_idx, speaker_idx] / np.sum(conf_matrix[speaker_idx, :])
    accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
    TP = conf_matrix[speaker_idx, speaker_idx]
    FP = np.sum(conf_matrix[:, speaker_idx]) - TP
    TN = np.trace(conf_matrix) - TP
    FN = np.sum(conf_matrix[speaker_idx, :]) - TP
    print("Speaker {}: Precision = {:.2f}, Recall = {:.2f}, Accuracy = {:.2f}".format(speaker, precision, recall, accuracy))
    print(likelihoods[:1])
    print("Correct/Total: {}/{}".format(correct, total))
    print("True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}".format(TP, FP, TN, FN))

    # Calculate the precision and accuracy of the model
    precision = ((np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))*100)
    accuracy = ((np.diag(conf_matrix).sum() / conf_matrix.sum())*100)
    print("Overall Precision = {:.2f}, Accuracy = {:.2f}".format(precision, accuracy))

# Print the precision and accuracy
print("Total Precision: {:.2f}".format(precision))
print("Total Accuracy: {:.2f}".format(accuracy))

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot the pre-emphasized signal
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Pre-Emphasized Signal')
plt.show()

# Plot the framed signal
plt.figure(figsize=(10, 4))
plt.plot(signal)
for i in range(num_frames):
    start = int(i*win_shift*rate)
    end = int(i*win_shift*rate+win_size*rate)
    plt.plot([start, start], [-1, 1], 'r')
    plt.plot([end, end], [-1, 1], 'r')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Framed Signal')
plt.show()

# Plot the Hamming window
plt.figure(figsize=(10, 4))
plt.plot(window)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Hamming Window')
plt.show()

# Plot the MFCC features
plt.figure(figsize=(10, 4))
plt.imshow(features.T, interpolation='nearest', origin='lower', aspect='auto')
plt.xlabel('Frame')
plt.ylabel('MFCC Coefficient')
plt.colorbar()
plt.title('MFCC Features for Test File')
plt.show()

# Plot Mel filterbank
plt.figure(figsize=(15, 5))
plt.title('Mel filterbank')
plt.plot(filterbank.T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filterbank magnitude')
plt.show()
