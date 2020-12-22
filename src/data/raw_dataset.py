"""This script generates raw data for further analysis and processing."""

import os
import yaml

from data_fetcher import SpotifyDataFetcher
from dataframes_merger import merge_playlist_dataframes


def main():
    with open('../configs/config.yaml', mode='r') as config_file:
        conf = yaml.load(config_file, Loader=yaml.FullLoader)

    id = os.environ['CLIENT_ID']
    secret = os.environ['CLIENT_SECRET']

    username = conf['api']['username']
    scope = conf['api']['scope']
    uri = conf['api']['uri']

    like = conf['playlists']['like']
    dislike = conf['playlists']['dislike']

    raw_path = conf['data']['raw-dataset']

    sdf = SpotifyDataFetcher(id, secret, username, scope, uri)
    df = merge_playlist_dataframes(sdf, username, like, dislike)

    df.to_csv(raw_path, index=True)


if __name__ == '__main__':
    main()
