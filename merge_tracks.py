import pandas as pd
import os
import urllib.parse
OUTPUT_DIR = os.path.join('db', 'src', 'main', 'resources', 'data', 'fingerprints')

def transform_image_url(url):
    if not isinstance(url, str) or not url:
        return url
    if 'freemusicarchive.org/file/' in url:
        path = url.split('/file/')[1]
        encoded_path = urllib.parse.quote(path, safe='')
        new_url = f"https://freemusicarchive.org/image/?file={encoded_path}&width=290&height=290&type=album"
        return new_url
    return url

def merge_tracks(tracks_csv, raw_tracks_csv, output_csv):
    tracks = pd.read_csv(tracks_csv, header=[0,1], index_col=0)
    raw_tracks = pd.read_csv(raw_tracks_csv, dtype=str)
    raw_tracks = raw_tracks.loc[:, ~raw_tracks.columns.str.contains('^Unnamed')]

    if isinstance(tracks.columns, pd.MultiIndex):
        album_id_col = ('album', 'id')
        song_name_col = ('track', 'title')
        album_col = ('album', 'title')
        artist_col = ('artist', 'name')
        genre_col = ('track', 'genre_top')
        track_id_series = tracks.index.astype(str)

    else:
        album_id_col = 'album.id'
        song_name_col = 'track.title'
        album_col = 'album.title'
        artist_col = 'artist.name'
        genre_col = 'track.genre_top'
        track_id_series = tracks['track_id'].astype(str)

    df = pd.DataFrame({
        'song_id': tracks.index,
        'song_name': tracks[song_name_col],
        'album': tracks[album_col],
        'artist': tracks[artist_col],
        'genre': tracks[genre_col],
        'album_id': tracks[album_id_col].astype(str),
        'track_id': track_id_series
    })

    df.fillna('', inplace=True)
    raw_tracks['album_id'] = raw_tracks['album_id'].astype(str)
    raw_tracks['track_id'] = raw_tracks['track_id'].astype(str)
    df_reset = df.reset_index(drop=True)
    raw_tracks_subset = raw_tracks[['album_id', 'track_id', 'track_image_file']].reset_index(drop=True)

    # This line transforms the image URLs to the correct format for the application,
    #     ensuring that all image links are consistent and accessible.
    
    raw_tracks_subset['track_image_file'] = raw_tracks_subset['track_image_file'].apply(transform_image_url)
    merged = pd.merge(
        df_reset,
        raw_tracks_subset,
        on=['album_id', 'track_id'],
        how='left'
    )
    merged['file_location'] = merged['track_id'].apply(
        lambda tid: os.path.join(OUTPUT_DIR, f"fingerprint_{tid}.csv")
    )
    merged.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to {output_csv}")

if __name__ == "__main__":
    tracks_csv = os.path.join("data", "fma_metadata", "tracks.csv")
    raw_tracks_csv = os.path.join("data", "fma_metadata", "raw_tracks.csv")
    output_csv = os.path.join("data", "fma_metadata", "tracks_merged.csv")
    merge_tracks(tracks_csv, raw_tracks_csv, output_csv)
