import os
import hashlib
import numpy as np
import pandas as pd
import librosa
import utils

AUDIO_DIR = os.path.join('data', 'fma_small')
OUTPUT_DIR = os.path.join('db', 'src', 'main', 'resources', 'data', 'fingerprints')

def load_csv(filepath):
    """
    Load a CSV file into a pandas DataFrame.
    """
    print(f"Loading {filepath}...")
    try:
        return pd.read_csv(filepath, index_col=0, low_memory=False)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

tracks = load_csv('data/fma_metadata/tracks_merged.csv')

n_fft = 2048
hop_length = 512
n_peaks = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_peaks(spectrogram, frequencies, sr, n_peaks=5):
    peaks = []
    for i in range(spectrogram.shape[1]):
        frame = spectrogram[:, i]
        peak_indices = np.argsort(frame)[-n_peaks:]
        peak_freqs = frequencies[peak_indices]
        peak_magnitudes = frame[peak_indices]
        timestamp = i * hop_length / sr
        peaks.append((timestamp, peak_freqs, peak_magnitudes))
    return peaks

def create_fingerprints(peaks, fan_out=15):
    fingerprints = []
    for i in range(len(peaks) - fan_out):
        anchor_time, anchor_freqs, anchor_mags = peaks[i]
        anchor_freq = anchor_freqs[np.argmax(anchor_mags)]
        for j in range(1, fan_out + 1):
            target_time, target_freqs, target_mags = peaks[i + j]
            target_freq = target_freqs[np.argmax(target_mags)]
            time_delta = target_time - anchor_time
            anchor_freq_int = int(anchor_freq)
            target_freq_int = int(target_freq)
            time_delta_int = int(time_delta * 1000)
            hash_str = f"{anchor_freq_int}|{target_freq_int}|{time_delta_int}"
            hash_value = hashlib.sha1(hash_str.encode()).hexdigest()
            fingerprints.append((hash_value, (anchor_time, anchor_freq_int)))
    return fingerprints

def fingerprint_song(track_id):
    # This function generates fingerprints for a song by extracting spectral peaks and hashing them.
    #   The process is similar to the Shazam algorithm, which pairs anchor and target peaks.
    audio_path = utils.get_audio_path(AUDIO_DIR, track_id)
    x, sr = librosa.load(audio_path, sr=None, mono=True)
    print('File: {}'.format(audio_path))
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks = find_peaks(stft, frequencies, sr, n_peaks=n_peaks)
    fingerprints = create_fingerprints(peaks)
    return fingerprints

def save_fingerprints_to_csv(track_id, fingerprints):
    fingerprint_data = {
        'hash': [],
        'timestamp': []
    }
    for hash_value, (timestamp, _) in fingerprints:
        fingerprint_data['hash'].append(hash_value)
        fingerprint_data['timestamp'].append(timestamp)
    df = pd.DataFrame(fingerprint_data)
    output_path = os.path.join(OUTPUT_DIR, f"fingerprint_{track_id}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(fingerprints)} fingerprints for track {track_id} to {output_path}")

def process_songs(track_ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for track_id in track_ids:
        try:
            fingerprints = fingerprint_song(track_id)
            save_fingerprints_to_csv(track_id, fingerprints)
            print(f"Successfully processed song {track_id}")
        except Exception as e:
            print(f"Error processing song {track_id}: {e}")

if __name__ == "__main__":
    track_ids = tracks.index[:70]
    process_songs(track_ids)

