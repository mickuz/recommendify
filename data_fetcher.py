import spotipy
from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd


class SpotifyAuthenticator:
    def __init__(self, client_id, client_secret):
        auth = SpotifyClientCredentials(client_id=client_id,
                                        client_secret=client_secret)

        self.sp = spotipy.Spotify(auth_manager=auth)


class SpotifyDataFetcher(SpotifyAuthenticator):
    def __init__(self, client_id, client_secret, username, scope, uri):
        super().__init__(client_id, client_secret)

        token = util.prompt_for_user_token(username=username,
                                           scope=scope,
                                           client_id=client_id,
                                           client_secret=client_secret,
                                           redirect_uri=uri)

        if token:
            self.sp = spotipy.Spotify(auth=token)
            print('Authorized!')
        else:
            print('Can\'t get a token!')

    def fetch_playlist_track_ids(self, username, playlist_name):
        playlists = self.sp.user_playlists(user=username)

        playlist_id = None
        for playlist in playlists['items']:
            if playlist['name'] == playlist_name:
                playlist_id = playlist['id']

        playlist_tracks = self.sp.playlist_tracks(playlist_id=playlist_id)
        pages_number = playlist_tracks['total'] // playlist_tracks['limit'] + 1

        track_ids = {}
        for _ in range(pages_number):
            for playlist_track in playlist_tracks['items']:
                track_name = playlist_track['track']['name']
                track_ids[track_name] = playlist_track['track']['id']
            playlist_tracks = self.sp.next(playlist_tracks)

        return track_ids

    def fetch_track_features(self, track_id):
        return self.sp.audio_features(tracks=track_id).pop()

    def create_dataframe(self, username, playlist_name):
        tracks = self.fetch_playlist_track_ids(username, playlist_name)
        track_infos = {name: self.fetch_track_features(id) for name, id in tracks.items()}

        tracks_dataframe = pd.DataFrame.from_dict(track_infos, orient='index')

        return tracks_dataframe
