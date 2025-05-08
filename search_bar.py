import os
import pandas as pd
import pickle
import re
import glob

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.track_info = []

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, track_info):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.track_info.append(track_info)
    
    def search(self, prefix):
        node = self.root
        result = []
        for char in prefix:
            if char not in node.children:
                return result
            node = node.children[char]
        self._dfs(node, prefix, result)
        return result
    
    def _dfs(self, node, prefix, result):
        if node.is_end_of_word:
            for info in node.track_info:
                result.append((prefix, info))
        for char, child_node in node.children.items():
            self._dfs(child_node, prefix + char, result)

def main():
    AUDIO_DIR = os.path.join('backend', 'fma_metadata')
    BASE_DIR = os.path.join('backend', 'data', 'fingerprints')
    SAVE_DIR = os.path.join('backend', 'data', 'index')

    os.makedirs(SAVE_DIR, exist_ok=True)

    tracks_csv_path = os.path.join(AUDIO_DIR, 'tracks_merged.csv')
    try:
        tracks_df = pd.read_csv(tracks_csv_path)
        print(f"Loaded {len(tracks_df)} tracks from CSV.")
    except FileNotFoundError:
        print(f"Error: Could not find tracks CSV at {tracks_csv_path}")
        return

    fingerprint_pattern = os.path.join(BASE_DIR, 'fingerprint_*.csv')
    fingerprint_files = glob.glob(fingerprint_pattern)
    print(f"Found {len(fingerprint_files)} fingerprint files.")

    track_id_pattern = re.compile(r'fingerprint_(\d+)\.csv')
    track_ids_with_fingerprints = []

    for filepath in fingerprint_files:
        filename = os.path.basename(filepath)
        match = track_id_pattern.match(filename)
        if match:
            track_id = match.group(1)
            track_ids_with_fingerprints.append(track_id)

    print(f"Extracted {len(track_ids_with_fingerprints)} valid track IDs from filenames.")

    song_trie = Trie()
    songs_added = 0

    for track_id in track_ids_with_fingerprints:
        # Ensure track_id is the same type as in the DataFrame
        try:
            track_id_int = int(track_id)
        except ValueError:
            continue  # skip if not a valid integer
        track_rows = tracks_df[tracks_df['track_id'] == track_id_int]
        if not track_rows.empty:
            for _, row in track_rows.iterrows():
                song_name = row['song_name']
                if pd.notna(song_name) and song_name.strip():
                    # This dictionary contains all relevant information about the track,
                    #     including song id, name, album, artist, genre, album id, track id,
                    #     track image file, and file location.
                    track_info = {
                        'song_id': row['song_id'],
                        'song_name': song_name,
                        'album': row['album'],
                        'artist': row['artist'],
                        'genre': row['genre'],
                        'album_id': row['album_id'],
                        'track_id': row['track_id'],
                        'track_image_file': row['track_image_file'],
                        'file_location': row['file_location']
                    }
                    song_trie.insert(song_name.lower(), track_info)
                    songs_added += 1

    print(f"Added {songs_added} songs to the Trie.")

    pickle_path = os.path.join(SAVE_DIR, 'song_trie.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(song_trie, f)

    print(f"Saved Trie to {pickle_path}")

    if songs_added > 0:
        sample_prefix = "a"
        results = song_trie.search(sample_prefix)

        print(f"\nSample search for prefix '{sample_prefix}':")
        for i, (song_name, info) in enumerate(results[:5], 1):
            print(f"{i}. {song_name} by {info['artist']}")

        if len(results) > 5:
            print(f"...and {len(results) - 5} more results")

if __name__ == "__main__":
    main()