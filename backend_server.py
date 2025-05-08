from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import pandas as pd
import pickle
import random
import time
from collections import defaultdict
from search_bar import Trie, TrieNode
import heapq
import numpy as np
import traceback

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
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            return bin(int1 ^ int2).count('1')
        except Exception as e:
            print(f"Error in hamming_distance: {str(e)}")
            return 9999
    
    def insert(self, hash_value, track_info):
        try:
            if self.root is None:
                self.root = {
                    'hash': hash_value,
                    'children': {},
                    'tracks': [track_info]
                }
                self.size += 1
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
                    self.size += 1
                    return
                    
                node = node['children'][distance]
        except Exception as e:
            print(f"Error in insert: {str(e)}")

    def _search_node(self, node, hash_value, max_distance, results, nodes_explored, depth):
        nodes_explored[0] += 1
        try:
            distance = self.hamming_distance(hash_value, node['hash'])
            if distance <= max_distance:
                for track_info in node['tracks']:
                    results.append((node['hash'], track_info, distance))
            for d in range(max(0, distance - max_distance), distance + max_distance + 1):
                if d in node['children']:
                    self._search_node(node['children'][d], hash_value, \
                                      max_distance, results, nodes_explored, depth + 1)
        except Exception as e:
            print(f"Error in search_node: {str(e)}")

    def search(self, hash_value, max_distance=3):
        if self.root is None:
            return []
        
        results = []
        nodes_explored = [0]
        try:
            self._search_node(self.root, hash_value, max_distance,\
                               results, nodes_explored, 0)
            if len(results) == 0 and random.random() < 0.1:
                print(f"No results found for hash {hash_value[:8]}" +
                      "... with distance ={max_distance}")
                print(f"Explored {nodes_explored[0]} nodes")
        except Exception as e:
            print(f"Error in search: {str(e)}")
        
        return results

def get_frequency_band(frequency):
    for band_id, (min_freq, max_freq) in enumerate(FREQUENCY_BANDS):
        if min_freq <= frequency < max_freq:
            return band_id
    return len(FREQUENCY_BANDS) - 1

def hamming_distance_int(int1, int2):
        return bin(int1 ^ int2).count('1')

def select_spread_vantage_points(fingerprints, num_points=20, sample_size=1000):
    if len(fingerprints) > sample_size:
        sample = random.sample(fingerprints, sample_size)
    else:
        sample = fingerprints.copy()
    sample_ints = {h: int(h, 16) for h in sample}
    selected = []
    selected_ints = []

    if sample:
        first = random.choice(sample)
        selected.append(first)
        selected_ints.append(sample_ints[first])
        sample.remove(first)
        del sample_ints[first]

    while len(selected) < num_points and sample:
        best_candidate = None
        max_min_distance = -1
        candidates = random.sample(sample, min(100, len(sample)))
        for candidate in candidates:
            candidate_int = sample_ints[candidate]
            min_distance = min(
                hamming_distance_int(candidate_int, sel_int)
                for sel_int in selected_ints
            )
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        if best_candidate:
            selected.append(best_candidate)
            selected_ints.append(sample_ints[best_candidate])
            sample.remove(best_candidate)
            del sample_ints[best_candidate]
        else:
            break
    return selected

def initialize_bk_trees():
    global bk_trees
    if bk_trees is None:
        bk_trees = [BKTree(band_id) for band_id in range(len(FREQUENCY_BANDS))]
        print("The BK-Trees for each frequency band have been initialized.")
        print("There is one tree per frequency band.")
    else:
        print("BK-Trees are already initialized.")
    return bk_trees

def process_fingerprint_file(track_id, file_path):
    global bk_trees, index_dirty
    if bk_trees is None:
        print("WARNING: BK-trees are not initialized")
        initialize_bk_trees()
    try:
        df = pd.read_csv(file_path)
        if 'anchor_freq' not in df.columns:
            return {"error": "Missing anchor_freq column"}
        band_counts = {band_id: 0 for band_id in range(len(FREQUENCY_BANDS))}
        inserted_count = 0
        for i, row in df.iterrows():
            hash_value = row['hash']
            timestamp = row['timestamp']
            anchor_freq = row['anchor_freq']
            band_id = get_frequency_band(anchor_freq)
            track_info = (track_id, timestamp, anchor_freq)
            if i < 5:
                before_size = bk_trees[band_id].size
                before_root = bk_trees[band_id].root is not None
            bk_trees[band_id].insert(hash_value, track_info)
            if i < 5:
                after_size = bk_trees[band_id].size
                after_root = bk_trees[band_id].root is not None
                if after_size > before_size or (not before_root and after_root):
                    print(f"Insert {i}: Successful, size: {before_size} -> {after_size}, root: {before_root} -> {after_root}")
                    inserted_count += 1
                else:
                    print(f"Insert {i}: Size unchanged ({before_size}), root: {before_root} -> {after_root}")
            band_counts[band_id] += 1
        index_dirty = True
        print("\nBK-tree status after processing:")
        for i, tree in enumerate(bk_trees):
            if tree.root is None:
                print(f"  Tree {i}: EMPTY (size={tree.size})")
            else:
                print(f"  Tree {i}: OK, size={tree.size}, has {len(tree.root['tracks'])} tracks at root")
        print(f"Band distribution: {band_counts}")
        return {
            "status": "success",
            "fingerprints_processed": len(df),
            "band_distribution": band_counts
        }
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred while processing the fingerprint file {file_path}:")
        print(f"{e}")
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
        print(f"The index was saved to {index_path}.")
        index_dirty = False
        return {"status": "success", "message": f"Index saved to {index_path}"}
    except Exception as e:
        print(f"An error occurred while saving the index:")
        print(f"{e}")
        return {"status": "error", "message": str(e)}

def load_index():
    global bk_trees, index_dirty
    try:
        index_path = os.path.join(INDEX_DIR, "bk_trees.pickle")
        if not os.path.exists(index_path):
            print("No existing index was found.")
            print("A new index will be initialized.")
            initialize_bk_trees()
            return {"status": "initialized", "message": "New index initialized"}
        with open(index_path, 'rb') as f:
            bk_trees = pickle.load(f)
        print(f"The index was loaded from {index_path}.")
        print("The index_dirty flag is now set to False.")
        index_dirty = False
        return {
            "status": "success",
            "message": f"Index loaded from {index_path}",
            "stats": get_index_stats()
        }
    except Exception as e:
        print(f"An error occurred while loading the index:")
        print(f"{e}")
        initialize_bk_trees()
        return {"status": "error", "message": f"Error: {str(e)}. New index initialized."}
    
def identify_song(query_fingerprints, max_distance=12, time_tolerance=3.0):
    global bk_trees
    if bk_trees is None:
        load_index()
        if bk_trees is None:
            return {"error": "Index not initialized"}
    print(f"Starting identification with {len(query_fingerprints)} fingerprints")
    print(f"Parameters: max_distance={max_distance}, time_tolerance={time_tolerance}")
    empty_trees = sum(1 for tree in bk_trees if tree.size == 0)
    if empty_trees == len(bk_trees):
        print("ERROR: All BK-trees are empty. No matches will be found.")
        return []
    banded_query = defaultdict(list)
    for hash_value, timestamp, anchor_freq in query_fingerprints:
        band_id = get_frequency_band(anchor_freq)
        banded_query[band_id].append((hash_value, timestamp, anchor_freq))
    print("Query fingerprints by band:")
    for band_id, fingerprints in banded_query.items():
        print(f"  Band {band_id}: {len(fingerprints)} fingerprints")
    track_matches = defaultdict(list)
    total_searched = 0
    total_matches = 0
    for band_id, fingerprints in banded_query.items():
        if band_id >= len(bk_trees):
            continue
        tree = bk_trees[band_id]
        if tree.size == 0:
            print(f"Skipping empty tree for band {band_id}")
            continue
        band_matches = 0
        for hash_value, query_time, _ in fingerprints:
            total_searched += 1
            similar_hashes = tree.search(hash_value, max_distance=max_distance)
            if not similar_hashes:
                continue
            if total_searched <= 5:
                print(f"Query {total_searched}: hash={hash_value[:8]}, found {len(similar_hashes)} similar hashes")
                for i, (db_hash, track_info, dist) in enumerate(similar_hashes[:3]):
                    print(f"  Match {i+1}: track={track_info[0]}, distance={dist}, time={track_info[1]}")
            per_track_best = {}
            for db_hash, (track_id, track_time, _), dist in similar_hashes:
                time_diff = abs(track_time - query_time)
                if (track_id not in per_track_best) or (time_diff < per_track_best[track_id][0]):
                    per_track_best[track_id] = (time_diff, query_time, track_time, dist)
            for track_id, match_info in per_track_best.items():
                track_matches[track_id].append(match_info)
                band_matches += 1
                total_matches += 1
        if band_matches > 0:
            print(f"Band {band_id}: found {band_matches} matches")
    print(f"Total searched: {total_searched}, matched: {total_matches}")
    results = []
    for track_id, matches in track_matches.items():
        close_matches = [m for m in matches if m[0] <= time_tolerance]
        score = len(close_matches) / len(query_fingerprints)
        avg_time_diff = sum(m[0] for m in matches) / len(matches) if matches else float('inf')
        min_time_diff = min(m[0] for m in matches) if matches else float('inf')
        avg_distance = sum(m[3] for m in matches) / len(matches) if matches else float('inf')
        result = {
            'track_id': track_id,
            'score': score,
            'close_matches': len(close_matches),
            'total_matches': len(matches),
            'avg_time_diff': avg_time_diff,
            'min_time_diff': min_time_diff,
            'avg_distance': avg_distance,
            'total_query_fingerprints': len(query_fingerprints)
        }
        results.append(result)
    results.sort(key=lambda x: (-x['score'], x['avg_time_diff']))
    print("\nTop results:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. Track: {result['track_id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Close matches: {result['close_matches']}/{result['total_query_fingerprints']} ({result['close_matches']/result['total_query_fingerprints']*100:.2f}%)")
        print(f"   Avg time diff: {result['avg_time_diff']:.4f}s")
        print(f"   Min time diff: {result['min_time_diff']:.4f}s")
        print(f"   Avg distance: {result.get('avg_distance', 'N/A')}")
    if results:
        return results[:1]
    return []

def rebuild_index_from_fingerprint_files(vantage_point_selection=True):
    global bk_trees, index_dirty
    start_time = time.time()
    initialize_bk_trees()
    fingerprint_files = [f for f in os.listdir(FINGERPRINTS_DIR) if f.endswith('.csv')]
    print(f"{len(fingerprint_files)} fingerprint files were found to process.")
    processed_count = 0
    error_count = 0
    fingerprint_count = 0
    if vantage_point_selection:
        band_fingerprints = [[] for _ in range(len(FREQUENCY_BANDS))]
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
                print(f"An error occurred while sampling file {filename}:")
                print(f"{e}")

        # Vantage points are being selected for each band
        # This maximizes the spread in the hash space
        vantage_points = []
        for band_id, fingerprints in enumerate(band_fingerprints):
            num_points = min(20, len(fingerprints))
            if num_points > 0:
                band_vantage_points = select_spread_vantage_points(fingerprints, num_points)
                vantage_points.append(band_vantage_points)
                print(f"{len(band_vantage_points)} vantage points were selected for band {band_id}.")
                print("These will be inserted first.")
            else:
                vantage_points.append([])
        for band_id, points in enumerate(vantage_points):
            for hash_value in points:
                bk_trees[band_id].insert(hash_value, ("vantage_point", 0, 0))
        # Now all the fingerprint files are being processed
        # Each fingerprint is added to the appropriate BK-Tree
    for filename in fingerprint_files:
        try:
            filepath = os.path.join(FINGERPRINTS_DIR, filename)
            track_id = filename.replace('fingerprint_', '').replace('.csv', '')
            df = pd.read_csv(filepath)
            if 'anchor_freq' not in df.columns:
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
            print(f"An error occurred while processing file {filename}:")
            print(f"{e}")
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
        temp_path = os.path.join(FINGERPRINTS_DIR, f"temp_{track_id}.csv")
        file.save(temp_path)
        try:
            df = pd.read_csv(temp_path)
            original_count = len(df)
            if 'anchor_freq' not in df.columns:
                anchor_freq_data = request.form.get('anchor_freq_data')
                if anchor_freq_data:
                    anchor_freqs = pd.read_json(anchor_freq_data)
                    if len(anchor_freqs) == len(df):
                        df['anchor_freq'] = anchor_freqs
                    else:
                        os.remove(temp_path)
                        return jsonify({
                            'status': 'error',
                            'message': f'Anchor frequency data size mismatch: {len(anchor_freqs)} vs {len(df)}'
                        }), 400
                else:
                    os.remove(temp_path)
                    return jsonify({
                        'status': 'error',
                        'message': 'Missing anchor_freq_data parameter'
                    }), 400
            MAX_FINGERPRINTS = 1000
            if len(df) > MAX_FINGERPRINTS:
                print(f"Reducing fingerprints from {len(df)} to {MAX_FINGERPRINTS}")
                df['band'] = df['anchor_freq'].apply(get_frequency_band)
                bands = df['band'].value_counts().to_dict()
                print(f"Original band distribution: {bands}")
                total_fingerprints = len(df)
                band_allocation = {}
                mid_band = 1
                if mid_band in bands:
                    mid_alloc = min(bands[mid_band], int(MAX_FINGERPRINTS * 0.5))
                    band_allocation[mid_band] = mid_alloc
                    remaining = MAX_FINGERPRINTS - mid_alloc
                else:
                    remaining = MAX_FINGERPRINTS
                low_band = 0
                if low_band in bands:
                    low_alloc = min(bands[low_band], int(MAX_FINGERPRINTS * 0.3))
                    band_allocation[low_band] = low_alloc
                    remaining -= low_alloc
                other_bands = [b for b in bands.keys() if b not in [0, 1]]
                other_total = sum(bands[b] for b in other_bands)
                if other_total > 0:
                    for band in other_bands:
                        proportion = bands[band] / other_total
                        band_allocation[band] = min(bands[band], 
                                                   max(1, int(remaining * proportion)))
                total_allocated = sum(band_allocation.values())
                if total_allocated > MAX_FINGERPRINTS:
                    max_band = max(band_allocation, key=band_allocation.get)
                    band_allocation[max_band] -= (total_allocated - MAX_FINGERPRINTS)
                total_allocated = sum(band_allocation.values())
                if total_allocated < MAX_FINGERPRINTS and mid_band in bands:
                    band_allocation[mid_band] = min(
                        bands[mid_band], 
                        band_allocation.get(mid_band, 0) + (MAX_FINGERPRINTS - total_allocated)
                    )
                print(f"Band allocation: {band_allocation}")
                selected_frames = []
                for band, count in band_allocation.items():
                    band_df = df[df['band'] == band]
                    if len(band_df) <= count:
                        selected_frames.append(band_df)
                    else:
                        band_df = band_df.sort_values('timestamp')
                        indices = np.linspace(0, len(band_df) - 1, count).astype(int)
                        selected_frames.append(band_df.iloc[indices])
                df = pd.concat(selected_frames)
                new_bands = df['band'].value_counts().to_dict()
                print(f"Final band distribution: {new_bands}")
                df = df.drop('band', axis=1)
            output_path = os.path.join(FINGERPRINTS_DIR, f"fingerprint_{track_id}.csv")
            df.to_csv(output_path, index=False)
            os.remove(temp_path)
            process_result = process_fingerprint_file(track_id, output_path)
            global index_dirty
            if index_dirty and random.random() < 0.1:
                save_index()
            return jsonify({
                'status': 'success',
                'message': f'Saved and processed {len(df)} out of {original_count} fingerprints for track {track_id}',
                'output_path': output_path,
                'reduced_ratio': f"{len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)",
                'process_result': process_result
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'File saved but processing failed: {str(e)}'
            }), 500
    except Exception as e:
        traceback.print_exc()
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
        print("Search error: Missing search query")
        return jsonify({
            'status': 'error',
            'message': 'Missing search query'
        }), 400
    trie_path = os.path.join('backend', 'data', 'index', 'song_trie.pickle')
    try:
        if not os.path.exists(trie_path):
            print("Search error: Song database not found.")
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
        print(f"Search success: {len(formatted_results)} results for query '{query}'")
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
    print("Loading index...")
    result = load_index()
    if result['status'] != 'success':
        print("No valid index was found.")
    else:
        empty_trees = sum(1 for tree in bk_trees if tree.size == 0)
        if empty_trees == len(bk_trees):
            print("WARNING: All BK-trees are empty. Index is not usable.")
    print("The server initialization is complete!")

initialize_server()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)