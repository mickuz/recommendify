import spotipy
from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd


class SpotifyAuthenticator:
    """Authorizes an application to access the Spotify Platform.

    Attributes
    ----------
    sp : spotipy.client.Spotify
        The Spotify object for accessing the Web API.
    """

    def __init__(self, client_id, client_secret):
        """
        Parameters
        ----------
        client_id : str
            The unique identifier of the application.
        client_secret : str
            The key passed in secure calls to the Spotify Accounts and
            Web API services.
        """
        auth = SpotifyClientCredentials(client_id=client_id,
                                        client_secret=client_secret)

        self.sp = spotipy.Spotify(auth_manager=auth)


class SpotifyDataFetcher(SpotifyAuthenticator):
    """Obtains the requested data from Spotify.

    Attributes
    ----------
    sp : spotipy.client.Spotify
        The Spotify object for accessing the Web API.

    Methods
    -------
    fetch_playlist_track_ids(username, playlist_name)
        Get ID for every track in a playlist.
    fetch_track_features(track_id):
        Get audio feature information for a single track.
    create_dataframe(username, playlist_name)
        Build a data frame with audio features for every track in a
        playlist.
    """

    def __init__(self, client_id, client_secret, username, scope, uri):
        """
        Parameters
        ----------
        client_id : str
            The unique identifier of the application.
        client_secret : str
            The key passed in secure calls to the Spotify Accounts and
            Web API services.
        username : str
            The name of a user asked to authorize an access.
        scope : str or None
            A space-separated list of scopes. If no scopes are
            specified, authorization will be granted only to access
            publicly available information.
        uri : str
            The URI to redirect to after the user grants or denies
            permission.
        """
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
        """Get ID for every track in a playlist.

        Parameters
        ----------
        username : str
            The name of a user who owns a given playlist.
        playlist_name : str
            The name of a playlist to retrieve the tracks from.

        Returns
        -------
        dict of {str : str}
            A dictionary containing names of the tracks and their
            respective IDs in a given playlist.
        """
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
        """Get audio feature information for a single track.

        Parameters
        ----------
        track_id : str
            The ID of a track to retrieve the features from.

        Returns
        -------
        dict of {str : int, float or str}
            A dictionary containing name of a given track and its
            respective audio features.
        """
        return self.sp.audio_features(tracks=track_id).pop()

    def create_dataframe(self, username, playlist_name):
        """Build a data frame with audio features for every track in a
        playlist.

        Parameters
        ----------
        username : str
            The name of a user who owns a given playlist.
        playlist_name : str
            The name of a playlist to fetch the tracks from.

        Returns
        -------
        pandas.core.frame.DataFrame
            The data frame containing names of the songs in a given
            playlist and their respective audio features.
        """
        tracks = self.fetch_playlist_track_ids(username, playlist_name)
        track_infos = {name: self.fetch_track_features(id)
                       for name, id in tracks.items()}

        tracks_dataframe = pd.DataFrame.from_dict(track_infos, orient='index')

        return tracks_dataframe
