from utils import merge_playlist_dataframes
from data_fetcher import SpotifyDataFetcher
from config import username, scope, redirect_uri, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET

import pandas as pd


def main():
    sdf = SpotifyDataFetcher(SPOTIPY_CLIENT_ID,
                             SPOTIPY_CLIENT_SECRET,
                             username,
                             scope,
                             redirect_uri)

    df = merge_playlist_dataframes(sdf,
                                   username,
                                   'Songs for Machine Learning project (good)',
                                   'Songs for Machine Learning project (bad)')

    df.to_csv('./songs.csv', index=True)


if __name__ == '__main__':
    main()
