from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import pandas as pd
import pickle
import random
import time
from collections import defaultdict

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.join('backend', 'data')
FINGERPRINTS_DIR = os.path.join(BASE_DIR, 'fingerprints')
INDEX_DIR = os.path.join(BASE_DIR, 'index')

for directory in [BASE_DIR, FINGERPRINTS_DIR, INDEX_DIR]:
    os.makedirs(directory, exist_ok=True)

FREQUENCY_BANDS = [
    (20, 500),
    (500, 2000),
    (2000, 8000),
    (8000, 20000)
]

bk_trees = None
index_dirty = False

class BKTree:
    def __init__(self, band_id):
        self.root = None
        self.band_id = band_id
        self.size = 0

    def hamming_distance(self, hash1, hash2):
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash1) * 4)
        min_len = min(len(bin1), len(bin2))
        return sum(c1 != c2 for c1, c2 in zip(bin1[:min_len], bin2[:min_len]))

    def insert(self, hash_value, track_info):
        self.size += 1
        if self.root is None:
            self.root = {
                'hash': hash_value,
                'children': {},
                'tracks': [track_info]
            }
            return
        node = self.root
        while True:
            distance = self.hamming_distance(hash_value, node['hash'])
            if distance == 0:
                if track_info not in node['tracks']:
                    node['tracks'].append(track_info)
                return
            if distance not in node['children']:
                node['children'][distance] = {
                    'hash': hash_value,
                    'children': {},
                    'tracks': [track_info]
                }
                return
            node = node['children'][distance]

    def search(self, hash_value, max_distance=2):
        if self.root is None:
            return []
        results = []
        def search_node(node, dist_so_far):
            distance = self.hamming_distance(hash_value, node['hash'])
            if distance <= max_distance:
                for track_info in node['tracks']:
                    results.append((node['hash'], track_info, distance))
            for d in range(distance - max_distance, distance + max_distance + 1):
                if d in node['children']:
                    search_node(node['children'][d], dist_so_far + 1)
        search_node(self.root, 0)
        return results

def get_frequency_band(frequency):
    for band_id, (min_freq, max_freq) in enumerate(FREQUENCY_BANDS):
        if min_freq <= frequency < max_freq:
            return band_id
    return len(FREQUENCY_BANDS) - 1

def select_spread_vantage_points(fingerprints, num_points=20, sample_size=1000):
    def hamming_distance(hash1, hash2):
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash1) * 4)
        min_len = min(len(bin1), len(bin2))
        return sum(c1 != c2 for c1, c2 in zip(bin1[:min_len], bin2[:min_len]))
    if len(fingerprints) > sample_size:
        sample = random.sample(fingerprints, sample_size)
    else:
        sample = fingerprints.copy()
    selected = []
    if sample:
        first = random.choice(sample)
        selected.append(first)
        sample.remove(first)
    while len(selected) < num_points and sample:
        best_candidate = None
        max_min_distance = -1
        candidates = random.sample(sample, min(100, len(sample)))

        for candidate in candidates:
            min_distance = min(hamming_distance(candidate, point) \
                               for point in selected)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        if best_candidate:
            selected.append(best_candidate)
            sample.remove(best_candidate)
        else:
            break
    return selected

def initialize_bk_trees():
    global bk_trees
    bk_trees = [BKTree(band_id) for band_id in range(len(FREQUENCY_BANDS))]
    print(
        "The BK-Trees for each frequency band have been initialized."
        "\nThere is one tree per frequency band."
    )
    return bk_trees

def process_fingerprint_file(track_id, file_path):
    global bk_trees, index_dirty
    if bk_trees is None:
        initialize_bk_trees()
    try:
        df = pd.read_csv(file_path)
        if 'anchor_freq' not in df.columns:
            return {"error": "Missing anchor_freq column"}
        band_counts = {band_id: 0 \
            for band_id in range(len(FREQUENCY_BANDS))}
        for i, row in df.iterrows():

            hash_value = row['hash']
            timestamp = row['timestamp']
            anchor_freq = row['anchor_freq']
            band_id = get_frequency_band(anchor_freq)
            track_info = (track_id, timestamp, anchor_freq)
            bk_trees[band_id].insert(hash_value, track_info)
            band_counts[band_id] += 1

        index_dirty = True
        print(f"Band distribution: {band_counts}")
        return {
            "status": "success",
            "fingerprints_processed": len(df),
            "band_distribution": band_counts
        }
    except Exception as e:
        print(
            f"An error occurred while processing the fingerprint file {file_path}:"
            f"\n{e}"
        )
        return {"error": str(e)}

def get_index_stats():
    if bk_trees is None:
        return {"status": "not_initialized"}
    stats = {
        "total_trees": len(bk_trees),
        "trees": []
    }
    for i, tree in enumerate(bk_trees):
        min_freq, max_freq = FREQUENCY_BANDS[i]
        stats["trees"].append({
            "band_id": i,
            "frequency_range": f"{min_freq}-{max_freq}Hz",
            "size": tree.size,
            "has_root": tree.root is not None
        })
    return stats

def save_index():
    global index_dirty
    if bk_trees is None:
        return {"status": "error", "message": "Index not initialized"}
    try:
        index_path = os.path.join(INDEX_DIR, "bk_trees.pickle")
        with open(index_path, 'wb') as f:
            pickle.dump(bk_trees, f)
        print(
            f"The index was saved to {index_path}."
            "\nThe index_dirty flag is now set to False."
        )
        index_dirty = False
        return {"status": "success", "message": f"Index saved to {index_path}"}
    except Exception as e:
        print(
            f"An error occurred while saving the index:"
            f"\n{e}"
        )
        return {"status": "error", "message": str(e)}

def load_index():
    global bk_trees, index_dirty
    try:
        index_path = os.path.join(INDEX_DIR, "bk_trees.pickle")
        if not os.path.exists(index_path):
            print(
                "No existing index was found."
                "\nA new index will be initialized."
            )
            initialize_bk_trees()
            return {"status": "initialized", "message": "New index initialized"}
        with open(index_path, 'rb') as f:
            bk_trees = pickle.load(f)
        
        #The index was loaded from {index_path}."
        # And The index_dirty flag is now set to False."
        index_dirty = False
        return {
            "status": "success",
            "message": f"Index loaded from {index_path}",
            "stats": get_index_stats()
        }
    except Exception as e:
        print(
            f"An error occurred while loading the index:"
            f"\n{e}"
        )
        initialize_bk_trees()
        return {"status": "error", "message": f"Error: {str(e)}. New index initialized."}

def identify_song(query_fingerprints):
    global bk_trees
    if bk_trees is None:
        load_index()
        if bk_trees is None:
            return {"error": "Index not initialized"}
    banded_query = defaultdict(list)
    for hash_value, timestamp, anchor_freq in query_fingerprints:
        band_id = get_frequency_band(anchor_freq)
        banded_query[band_id].append((hash_value, timestamp, anchor_freq))
    matches = defaultdict(list)
    for band_id, fingerprints in banded_query.items():
        if band_id >= len(bk_trees):
            continue
        tree = bk_trees[band_id]
        for hash_value, query_time, _ in fingerprints:
            similar_hashes = tree.search(hash_value, max_distance=2)
            for _, (track_id, track_time, _), distance in similar_hashes:
                matches[track_id].append((query_time, track_time))
    results = []
    for track_id, time_pairs in matches.items():
        if len(time_pairs) < 5:
            continue
        offsets = [track_time - query_time for query_time, track_time in time_pairs]
        offset_counts = defaultdict(int)
        tolerance = 0.005
        for offset in offsets:
            rounded_offset = round(offset / tolerance) * tolerance
            offset_counts[rounded_offset] += 1
        if not offset_counts:
            continue
        best_offset, count = max(offset_counts.items(), key=lambda x: x[1])
        confidence = count / len(query_fingerprints)
        if confidence > 0.05:
            results.append({
                'track_id': track_id,
                'confidence': confidence,
                'matched_count': count,
                'total_fingerprints': len(query_fingerprints),
                'time_offset': best_offset
            })
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:10]

def rebuild_index_from_fingerprint_files(vantage_point_selection=True):
    global bk_trees, index_dirty
    start_time = time.time()
    initialize_bk_trees()
    fingerprint_files = [f for f in os.listdir(FINGERPRINTS_DIR) if f.endswith('.csv')]
    print(
        f"{len(fingerprint_files)} fingerprint files were found to process."
        "\nThe index rebuild will now begin."
    )
    processed_count = 0
    error_count = 0
    fingerprint_count = 0
    if vantage_point_selection:
        band_fingerprints = [[] for _ in range(len(FREQUENCY_BANDS))]
        
        # The Sample fingerprints are being collected for vantage point selection."
        # This uses up to 100 files and 50 fingerprints per track."
        
        for filename in fingerprint_files[:min(100, len(fingerprint_files))]:
            try:
                filepath = os.path.join(FINGERPRINTS_DIR, filename)
                track_id = filename.replace('fingerprint_', '').replace('.csv', '')
                df = pd.read_csv(filepath)
                if 'anchor_freq' not in df.columns:
                    continue
                sample_size = min(50, len(df))
                for _, row in df.sample(sample_size).iterrows():
                    band_id = get_frequency_band(row['anchor_freq'])
                    band_fingerprints[band_id].append(row['hash'])
            except Exception as e:
                print(
                    f"An error occurred while sampling file {filename}:"
                    f"\n{e}"
                )
        
            # The Vantage points are being selected for each band.
            # This maximizes the spread in the hash space.
    
        vantage_points = []
        for band_id, fingerprints in enumerate(band_fingerprints):
            num_points = min(20, len(fingerprints))
            if num_points > 0:
                band_vantage_points = select_spread_vantage_points(fingerprints, num_points)
                vantage_points.append(band_vantage_points)
                print(
                    f"{len(band_vantage_points)} vantage points were selected for band {band_id}."
                    "\nThese will be inserted first."
                )
            else:
                vantage_points.append([])
        
        for band_id, points in enumerate(vantage_points):
            for hash_value in points:
                bk_trees[band_id].insert(hash_value, ("vantage_point", 0, 0))
    
        # Now that all the fingerprint files are being processed.
        # Each fingerprint is added to the appropriate BK-Tree.
    
    for filename in fingerprint_files:
        try:
            filepath = os.path.join(FINGERPRINTS_DIR, filename)
            track_id = filename.replace('fingerprint_', '').replace('.csv', '')
            df = pd.read_csv(filepath)
            if 'anchor_freq' not in df.columns:
                print(
                    f"{filename} is being skipped because it has no anchor_freq column."
                    "\nThis file will not be indexed."
                )
                continue
            band_counts = {band_id: 0 for band_id in range(len(FREQUENCY_BANDS))}
            for i, row in df.iterrows():

                hash_value = row['hash']
                timestamp = row['timestamp']
                anchor_freq = row['anchor_freq']
                band_id = get_frequency_band(anchor_freq)
                track_info = (track_id, timestamp, anchor_freq)
                bk_trees[band_id].insert(hash_value, track_info)
                band_counts[band_id] += 1
                fingerprint_count += 1

            processed_count += 1
           
        except Exception as e:
            print(
                f"An error occurred while processing file {filename}:"
                f"\n{e}"
            )
            error_count += 1
    index_dirty = True
    save_result = save_index()
    elapsed_time = time.time() - start_time
    result = {
        "status": "success",
        "files_processed": processed_count,
        "errors": error_count,
        "fingerprints_indexed": fingerprint_count,
        "elapsed_time": elapsed_time,
        "index_stats": get_index_stats(),
        "save_result": save_result
    }
    
    print(f"Processed {processed_count} files with {fingerprint_count} fingerprints")
    return result

@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({
        'status': 'success',
        'message': 'Connection successful from backend server!'
    })

@app.route('/api/save_fingerprints', methods=['POST'])
def save_fingerprints():
    try:
        if 'track_id' not in request.form:
            return jsonify({
                'status': 'error',
                'message': 'Missing track_id parameter'
            }), 400
        track_id = request.form['track_id']
        if 'fingerprint_csv' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Missing fingerprint_csv file'
            }), 400
        file = request.files['fingerprint_csv']
        os.makedirs(FINGERPRINTS_DIR, exist_ok=True)
        output_path = os.path.join(FINGERPRINTS_DIR, f"fingerprint_{track_id}.csv")
        file.save(output_path)
        try:
            df = pd.read_csv(output_path)
            if 'anchor_freq' not in df.columns:
                anchor_freq_data = request.form.get('anchor_freq_data')
                if anchor_freq_data:
                    anchor_freqs = pd.read_json(anchor_freq_data)
                    if len(anchor_freqs) == len(df):
                        df['anchor_freq'] = anchor_freqs
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': f'Anchor frequency data size mismatch: {len(anchor_freqs)} vs {len(df)}'
                        }), 400
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Missing anchor_freq_data parameter'
                    }), 400
                df.to_csv(output_path, index=False)
            process_result = process_fingerprint_file(track_id, output_path)
            global index_dirty
            if index_dirty and random.random() < 0.1:
                save_index()
            return jsonify({
                'status': 'success',
                'message': f'Saved and processed fingerprints CSV for track {track_id} with {len(df)} rows',
                'output_path': output_path,
                'process_result': process_result
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'File saved but processing failed: {str(e)}'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
        }), 500

@app.route('/api/index/stats', methods=['GET'])
def index_stats():
    stats = get_index_stats()
    return jsonify({
        'status': 'success',
        'stats': stats
    })

@app.route('/api/index/save', methods=['POST'])
def save_index_route():
    result = save_index()
    return jsonify(result)

@app.route('/api/index/load', methods=['POST'])
def load_index_route():
    result = load_index()
    return jsonify(result)

@app.route('/api/index/rebuild', methods=['POST'])
def rebuild_index_route():
    use_vantage_points = request.json.get('use_vantage_points', True)
    result = rebuild_index_from_fingerprint_files(use_vantage_points)
    return jsonify(result)

@app.route('/api/identify', methods=['POST'])
def identify_route():
    try:
        if not request.json or 'fingerprints' not in request.json:
            return jsonify({
                'status': 'error',
                'message': 'Missing fingerprints data'
            }), 400
        fingerprints = request.json['fingerprints']
        if not isinstance(fingerprints, list):
            return jsonify({
                'status': 'error',
                'message': 'Fingerprints must be a list'
            }), 400
        results = identify_song(fingerprints)
        return jsonify({
            'status': 'success',
            'matches': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
def search_songs():
    query = request.args.get('query', '').lower()
    
    if not query:
        return jsonify({
            'status': 'error',
            'message': 'Missing search query'
        }), 400
    
    trie_path = os.path.join('backend', 'data', 'structures', 'song_trie.pickle')
    
    try:
        # Load the song trie
        if not os.path.exists(trie_path):
            return jsonify({
                'status': 'error',
                'message': 'Song database not found. Please run search_bar.py first to build the song database.'
            }), 404
        
        with open(trie_path, 'rb') as f:
            song_trie = pickle.load(f)
        
        results = song_trie.search(query)
        
        results.sort(key=lambda x: len(x[0]))
        
        top_results = results[:5]
        
        formatted_results = []
        for song_name, track_info in top_results:
            formatted_results.append({
                'song_name': song_name,
                'track_id': track_info['track_id'],
                'artist': track_info['artist'],
                'album': track_info['album'],
                'genre': track_info['genre'],
                'file_location': track_info['file_location']
            })
        
        return jsonify({
            'status': 'success',
            'results': formatted_results
        })
    
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Search error: {str(e)}'
        }), 500


def initialize_server():
    global bk_trees
    result = load_index()
    if result['status'] != 'success':
        print("No valid index was found.")

    print("The server initialization is complete!")

initialize_server()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)