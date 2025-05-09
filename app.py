from flask import Flask, render_template, request, jsonify
import os
import requests
import numpy as np
import hashlib
import librosa
import sys
import traceback

sys.set_int_max_str_digits(100000)

app = Flask(__name__)

os.makedirs(os.path.join('static', 'css'), exist_ok=True)
os.makedirs(os.path.join('static', 'js'), exist_ok=True)

n_fft = 2048
hop_length = 512
n_peaks = 5

FREQUENCY_BANDS = [
    (20, 500),
    (500, 2000),
    (2000, 8000),
    (8000, 20000)
]

AUDIO_DIR = os.path.join('data', 'fma_small')
BACKEND_URL = "http://localhost:8080"

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
    fan_out = min(fan_out, 15)
    print(f"Creating fingerprints with {len(peaks)} peaks and fan_out={fan_out}")
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
                try:
                    anchor_freq_int = int(anchor_freq)
                    target_freq_int = int(target_freq)
                    time_delta_int = int(time_delta * 1000)
                except (ValueError, OverflowError) as e:
                    print(f"Error converting values: {e}")
                    print(f"anchor_freq={anchor_freq}, target_freq={target_freq}, time_delta={time_delta}")
                    continue
                hash_str = f"{anchor_freq_int}|{target_freq_int}|{time_delta_int}"
                hash_value = hashlib.sha1(hash_str.encode()).hexdigest()
                if not all(c in '0123456789abcdefABCDEF' for c in hash_value) or len(hash_value) != 40:
                    print(f"Invalid hash generated: {hash_value}")
                    continue
                fingerprint_data = (hash_value, anchor_time, anchor_freq_int)
                banded_fingerprints[band_id].append(fingerprint_data)
                all_fingerprints.append((hash_value, (anchor_time, anchor_freq_int)))
    total_fps = sum(len(fps) for fps in banded_fingerprints.values())
    print(f"Generated {total_fps} total fingerprints across {len(FREQUENCY_BANDS)} bands")
    for band_id, fps in banded_fingerprints.items():
        print(f"Band {band_id}: {len(fps)} fingerprints")
    return banded_fingerprints, all_fingerprints

def query_song(audio_file, duration=10):
    try:
        print(f"Loading audio file: {audio_file}")
        x, sr = librosa.load(audio_file, sr=None, mono=True)
        if duration:
            try:
                duration = float(duration)
                samples = min(int(duration * sr), len(x))
                x = x[:samples]
            except ValueError:
                print(f"Invalid duration value: {duration}, using full audio")
        print(f'Query: {audio_file}')
        print(f'Duration: {len(x) / sr:.2f}s, {len(x)} samples')
        print("Calculating STFT...")
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        print(f"STFT shape: {stft.shape}")
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        print(f"Frequency range: {frequencies[0]}-{frequencies[-1]} Hz")
        print("Finding peaks...")
        peaks = find_peaks(stft, frequencies, sr, n_peaks=n_peaks)
        print(f"Found {len(peaks)} peaks")
        print("Creating fingerprints...")
        banded_fingerprints, all_fingerprints = create_frequency_banded_fingerprints(peaks)
        query_fingerprints = []
        for band_id, band_fps in banded_fingerprints.items():
            for hash_value, timestamp, anchor_freq in band_fps:
                query_fingerprints.append([hash_value, timestamp, anchor_freq])
        print(f"Sending {len(query_fingerprints)} fingerprints to backend")
        if len(query_fingerprints) > 0:
            first_fp = query_fingerprints[0]
            print(f"Sample fingerprint: {first_fp}")
            print(f"Hash type: {type(first_fp[0])}, length: {len(first_fp[0])}")
        response = requests.post(
            f"{BACKEND_URL}/api/identify",
            json={'fingerprints': query_fingerprints}
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                if len(result) > 0:
                    print(f"Identification successful")
                    return result[0]
                else:
                    return {}
            return {}
        else:
            print(f"Error identifying song: {response.status_code}")
            try:
                error_text = response.text
                print(f"Error details: {error_text}")
                return {'status': 'error', 'message': error_text}
            except Exception as e:
                return {'status': 'error', 'message': f"Failed to parse error response: {str(e)}"}
    except Exception as e:
        print(f"Error in query_song: {e}")
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

@app.route('/')
def index():
    config = {
        'backend_url': 'http://localhost:8080',
        'duration': '10'
    }
    return render_template('identify.html', config=config)

@app.route('/api/identify', methods=['POST'])
def api_identify():
    print("Received request to identify song")
    try:
        print(f"Request content type: {request.content_type}")
        print(f"Request data: {request.get_data(as_text=True)}")
        data = request.get_json(silent=True)
        if data is None:
            print("Failed to parse JSON data")
            return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
        print(f"Parsed data: {data}")
        audio_file_path = data.get('audioFilePath', '')
        duration = data.get('duration', 10)
        print(f"Audio file path: '{audio_file_path}'")
        print(f"Duration: {duration}")
        if not audio_file_path:
            print("Missing audio file path")
            return jsonify({'status': 'error', 'message': 'audioFilePath is required'}), 400
        if not os.path.exists(audio_file_path):
            print(f"File not found: {audio_file_path}")
            alt_paths = [
                os.path.join(os.getcwd(), audio_file_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_file_path)
            ]
            for alt_path in alt_paths:
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    print(f"Found file at: {alt_path}")
                    audio_file_path = alt_path
                    break
            else:
                print("No alternative paths found")
                return jsonify({'status': 'error', 'message': f'File not found: {audio_file_path}'}), 404
        print(f"Querying song: {audio_file_path}")
        result = query_song(audio_file_path, duration)
        return jsonify(result)
    except Exception as e:
        print(f"Error in api_identify: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/test_identify', methods=['GET'])
def test_identify():
    """Test endpoint with known audio file"""
    try:
        test_dirs = [
            'data/fma_small',
            os.path.join(os.getcwd(), 'data/fma_small')
        ]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"Searching for MP3 files in {test_dir}")
                for root, dirs, files in os.walk(test_dir):
                    for file in files:
                        if file.endswith('.mp3'):
                            test_file = os.path.join(root, file)
                            print(f"Found test file: {test_file}")
                            result = query_song(test_file, 5)
                            return jsonify({
                                'status': 'success',
                                'test_file': test_file,
                                'result': result
                            })
        return jsonify({
            'status': 'error',
            'message': 'No test audio files found'
        }), 404
    except Exception as e:
        print(f"Error in test_identify: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='localhost')