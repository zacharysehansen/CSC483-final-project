import os
import hashlib
import numpy as np
import pandas as pd
import librosa
import requests
from tqdm import tqdm
import argparse

import utils

AUDIO_DIR = os.path.join('data', 'fma_small')
BACKEND_URL = "http://localhost:8080"

def load_csv(filepath):
    print(f"Loading {filepath}...")
    try:
        return pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

n_fft = 2048
hop_length = 512
n_peaks = 5

FREQUENCY_BANDS = [
    (20, 500),
    (500, 2000),
    (2000, 8000),
    (8000, 20000)
]

def get_frequency_band(frequency):
    for band_id, (min_freq, max_freq) in enumerate(FREQUENCY_BANDS):
        if min_freq <= frequency < max_freq:
            return band_id
    return len(FREQUENCY_BANDS) - 1

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

def create_frequency_banded_fingerprints(peaks, fan_out=15):
    banded_fingerprints = {band_id: [] for band_id in range(len(FREQUENCY_BANDS))}
    all_fingerprints = []

    for i in range(len(peaks) - fan_out):
        anchor_time, anchor_freqs, anchor_mags = peaks[i]
        for af_idx, anchor_freq in enumerate(anchor_freqs):
            anchor_mag = anchor_mags[af_idx]
            band_id = get_frequency_band(anchor_freq)

            for j in range(1, fan_out + 1):
                target_time, target_freqs, target_mags = peaks[i + j]
                target_idx = np.argmax(target_mags)
                target_freq = target_freqs[target_idx]
                time_delta = target_time - anchor_time

                anchor_freq_int = int(anchor_freq)
                target_freq_int = int(target_freq)
                time_delta_int = int(time_delta * 1000)

                hash_str = f"{anchor_freq_int}|{target_freq_int}|{time_delta_int}"
                hash_value = hashlib.sha1(hash_str.encode()).hexdigest()
                fingerprint_data = (hash_value, anchor_time, anchor_freq_int)
                banded_fingerprints[band_id].append(fingerprint_data)
                all_fingerprints.append((hash_value, (anchor_time, anchor_freq_int)))

    return banded_fingerprints, all_fingerprints

def fingerprint_song(track_id):

    audio_path = utils.get_audio_path(AUDIO_DIR, track_id)
    x, sr = librosa.load(audio_path, sr=None, mono=True)

    print('File: {}'.format(audio_path))
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks = find_peaks(stft, frequencies, sr, n_peaks=n_peaks)
    banded_fingerprints, all_fingerprints = create_frequency_banded_fingerprints(peaks)
    return banded_fingerprints, all_fingerprints

def create_csv_from_fingerprints(fingerprints):
    fingerprint_data = {
        'hash': [],
        'timestamp': [],
        'anchor_freq': []
    }
    for hash_value, (timestamp, anchor_freq) in fingerprints:
        fingerprint_data['hash'].append(hash_value)
        fingerprint_data['timestamp'].append(timestamp)
        fingerprint_data['anchor_freq'].append(anchor_freq)

    return pd.DataFrame(fingerprint_data)

def send_fingerprints_to_backend(track_id, fingerprints_df):
    try:
        csv_content = fingerprints_df.to_csv(index=False)
        from io import StringIO, BytesIO
        csv_buffer = BytesIO(csv_content.encode('utf-8'))
        files = {'fingerprint_csv': ('fingerprint.csv', csv_buffer, 'text/csv')}
        data = {'track_id': track_id}
        response = requests.post(
            f"{BACKEND_URL}/api/save_fingerprints",
            files=files,
            data=data
        )
        if response.status_code == 200:
            print(f"Successfully sent fingerprints CSV for track {track_id}")
            return response.json()
        else:
            print(f"Error sending fingerprints for track {track_id}: {response.text}")
            return {'status': 'error', 'message': response.text}
    except Exception as e:
        print(f"Error in send_fingerprints_to_backend: {e}")
        return {'status': 'error', 'message': str(e)}

def process_tracks(csv_file, limit=None):
    tracks_df = load_csv(csv_file)
    if tracks_df.empty:
        print(f"No tracks found in {csv_file}")
        return
    if limit:
        tracks_df = tracks_df.head(limit)
    print(f"Processing {len(tracks_df)} tracks...")

    success_count = 0
    error_count = 0
    for i, row in tqdm(tracks_df.iterrows(), total=len(tracks_df)):
        try:
            track_id = row['track_id']
            print(f"\nProcessing track {track_id}")
            j, all_fingerprints = fingerprint_song(track_id)
            fingerprints_df = create_csv_from_fingerprints(all_fingerprints)
            response = send_fingerprints_to_backend(track_id, fingerprints_df)
            if response.get('status') == 'success':
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Error processing track {row.get('track_id', 'unknown')}: {e}")
            error_count += 1
    print(f"\nFinished processing. Success: {success_count}, Errors: {error_count}")

def query_song(audio_file, duration=10):
    try:
        x, sr = librosa.load(audio_file, sr=None, mono=True)
        if duration:
            samples = min(int(duration * sr), len(x))
            x = x[:samples]
        print(f'Query: {audio_file}')
        print(f'Duration: {len(x) / sr:.2f}s, {len(x)} samples')
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        peaks = find_peaks(stft, frequencies, sr, n_peaks=n_peaks)
        banded_fingerprints, all_fingerprints = create_frequency_banded_fingerprints(peaks)
        query_fingerprints = []
        for band_id, band_fps in banded_fingerprints.items():
            for hash_value, timestamp, anchor_freq in band_fps:
                query_fingerprints.append([hash_value, timestamp, anchor_freq])
        response = requests.post(
            f"{BACKEND_URL}/api/identify",
            json={'fingerprints': query_fingerprints}
        )
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error identifying song: {response.text}")
            return {'status': 'error', 'message': response.text}
    except Exception as e:
        print(f"Error in query_song: {e}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio fingerprinting and identification')
    parser.add_argument('--process', help='Process tracks and save fingerprints', action='store_true')
    parser.add_argument('--identify', help='Path to audio file to identify')
    parser.add_argument('--limit', help='Limit number of tracks to process', type=int, default=None)
    parser.add_argument('--duration', help='Duration in seconds to analyze for identification', type=float, default=10)
    args = parser.parse_args()
    try:
        response = requests.get(f"{BACKEND_URL}/api/test")
        if response.status_code == 200:
            print("Backend connection successful!")
        else:
            print(f"Backend connection failed: {response.status_code} {response.text}")
            print("Make sure the backend server is running and accessible.")
            exit(1)
    except Exception as e:
        print(f"Backend connection error: {e}")
        print("Make sure the backend server is running and accessible.")
        exit(1)
    if args.process:
        merged_tracks_csv = os.path.join('data', 'fma_metadata', 'tracks_merged.csv')
        process_tracks(merged_tracks_csv, limit=args.limit)
        try:
            print("Rebuilding index on the backend server...")
            response = requests.post(
                f"{BACKEND_URL}/api/index/rebuild"
            )
            if response.status_code == 200:
                print("Successfully rebuilt the BK tree index on the backend")
            else:
                print(f"Error rebuilding BK tree index: Status {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Error connecting to backend to rebuild the BK tree index: {e}")
    if args.identify:
        result = query_song(args.identify, duration=args.duration)
        print("\nIdentification results:")
        if result.get('status') == 'success' and 'matches' in result:
            matches = result['matches']
            if matches:
                for i, match in enumerate(matches[:5]):
                    print(f"{i+1}. Track ID: {match['track_id']}")
                    print(f"   Confidence: {match['confidence']*100:.2f}%")
                    print(f"   Matched: {match['matched_count']}/{match['total_fingerprints']} fingerprints")
                    print(f"   Time offset: {match['time_offset']}s")
            else:
                print("No matches found")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")