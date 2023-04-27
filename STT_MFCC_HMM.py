import os
import numpy as np
import soundfile as sf
from mpmath import eps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm
from summa.summarizer import summarize
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import jiwer
import glob
import warnings
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")

rate = 16000
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
num_states = 25
# Create a list of Mel filterbank frequencies
mel_freqs = np.linspace(lower_freq, upper_freq, num_filterbanks + 2)
hz_freqs = mel_freqs
bin_width = np.floor((win_size * rate) / 2)
bin_idx = np.floor(hz_freqs / (rate / (2 * bin_width)))

# Load the Librispeech dataset
data_dir = 'thesisSpeechData/19'
labels = []
mfccs = []
transcripts = {}

# Load transcripts from specified directories
transcripts_dir1 = "thesisSpeechData/19/198"
transcripts_dir2 = "thesisSpeechData/19/227"
np.set_printoptions(threshold=np.inf)
total_words_Transcrip = 0

for transcripts_dir in [transcripts_dir1, transcripts_dir2]:
    for file_name in os.listdir(transcripts_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(transcripts_dir, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    transcript_id = parts[0]
                    transcript = ' '.join(parts[1:]).lower()
                    transcripts[transcript_id] = transcript
                    total_words_Transcrip += len(transcript.split())

print(f"Total number of words in transcript library: {total_words_Transcrip}")

# Load audio and extract MFCC features
for speaker in os.listdir(data_dir):
    speaker_dir = os.path.join(data_dir, speaker)
    for file_name in os.listdir(speaker_dir):
        if file_name.endswith('.flac'):
            file_path = os.path.join(speaker_dir, file_name)
            transcript_id = file_name.split('.')[0]
            if transcript_id in transcripts:
                label = transcripts[transcript_id]
                labels.append(label)
                signal, rate = sf.read(file_path, dtype='int16')
                # Apply pre-emphasis
                signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

                # Apply Hamming window
                window_size = int(win_size * rate)
                window = np.hamming(window_size)
                num_frames = int(np.ceil(len(signal) / rate / win_shift))
                pad_len = num_frames * int(win_shift * rate) + window_size - len(signal)
                signal = np.pad(signal, (0, pad_len), mode='constant')
                frames = np.zeros((num_frames, window_size))
                for i in range(num_frames):
                    start = int(i * win_shift * rate)
                    frames[i] = signal[start:start + window_size] * window

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
                features = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :num_states]
                mfccs.append(features)


# Convert labels to integers
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels = label_encoder.transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)

# Train an HMM for each label
models = []
for label in np.unique(y_train):
    model = hmm.GaussianHMM(n_components=15, covariance_type='full', min_covar=1e-3, algorithm="any")
    model.fit(np.vstack([x for i, x in enumerate(X_train) if y_train[i] == label]))
    models.append(model)

# Combine all transcript files into one corpus
corpus = []
for dirt in [transcripts_dir1, transcripts_dir2]:
    transcript_files = glob.glob(os.path.join(dirt, '*.trans.txt'))
    for file in transcript_files:
        with open(file, 'r') as f:
            for line in f:
                transcript = line.split(' ', 1)[1].strip().lower()
                corpus = corpus + transcript.split()

# Get the total number of words in the corpus
total_words = len(corpus)

# Predict labels and words for the test data
y_pred = []
recognized_words = []  # new array to store recognized words
for mfcc_feat in X_test:
    scores = [model.score(mfcc_feat) for model in models]
    y_pred.append(np.argmax(scores))
    word = label_encoder.inverse_transform(np.array(np.argmax(scores)).ravel())
    recognized_words.append(word)  # append recognized word to new array

# Calculate WER and error measures
ref = [corpus[i] for i in y_test]
hyp = [word[0] for word in recognized_words]
wer = jiwer.wer(ref, hyp)
measures = jiwer.compute_measures(ref, hyp)
# Print results
print(f'WER: {wer:.2f}')
print(f'S: {measures["substitutions"]}, I: {measures["insertions"]}, D: {measures["deletions"]}, N: {len(ref)}')

# Print the recognized words
print("Recognized words:")
for word in recognized_words:
    print(word, end=' ')
    print()  # add a newline character after printing each array

# Open the file in write mode
with open("recognized_words.txt", 'w') as f:
    # Iterate through the recognized words
    for word in recognized_words:
        # Write each word to a new line in the file
        f.write(str(word) + '\n')

# Read recognized words from text file
with open('recognized_words.txt', 'r') as f:
    recognized_words_str = f.read()

# Split the recognized text into sentences
sentences = recognized_words_str.split('. ')

# Keep track of the sentences that have already been summarized
summarized_sentences = set()

# Iterate through the sentences and remove those that have already been summarized
sentences_to_summarize = []
for sentence in sentences:
    if sentence not in summarized_sentences:
        summarized_sentences.add(sentence)
        sentences_to_summarize.append(sentence)

# Combine the remaining sentences into a string
recognized_words_str = '. '.join(sentences_to_summarize)

# Summarize the recognized words
summary = summarize(recognized_words_str, ratio=0.5)

# Print the summary
print("\nSummary:")
print(summary)

print("Signal",signal[:10])
print("Pre-Emphasis",pre_emphasis)
print("Hamming Window: ",window[:10])
print("Frames: ",frames[:10])
print("Mel Filterbank: ",filterbank.T[:10])
print("Features: ",features[:10])

# Print the acoustic model
print("\nAcoustic model:")
print(model.transmat_)

# Print the language model
print("Language model:")
for i in range(15):
    print(model.means_[i], model.covars_[i])

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
plt.imshow(features.T, origin='lower', aspect='auto', cmap='jet')
plt.title('MFCC Features')
plt.xlabel('Frame')
plt.ylabel('Coefficient')
plt.colorbar()
plt.show()

# Plot Mel filterbank
plt.figure(figsize=(15, 5))
plt.title('Mel filterbank')
plt.plot(filterbank.T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filterbank magnitude')
plt.show()
