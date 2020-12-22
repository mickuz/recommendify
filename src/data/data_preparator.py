"""This module implements functions preparing raw data frame."""

import pandas as pd


def merge_playlist_dataframes(sdf, username, playlist_like, playlist_dislike):
    """Combine two data frames with track features from playlists involving
    songs that were respectively liked and disliked by a user.

    Parameters
    ----------
    sdf : data_fetcher.SpotifyDataFetcher
        This object is assumed to load the data about the tracks in playlists.
    username : str
        The name of a user who owns given playlists.
    playlist_like : str
        The name of a playlist with tracks liked by the user.
    playlist_dislike : str
        The name of a playlist with tracks disliked by the user.

    Returns
    -------
    pandas.core.frame.DataFrame
        The result of concatenation of two given data frames.
    """
    df_like = sdf.create_dataframe(username, playlist_like)
    df_dislike = sdf.create_dataframe(username, playlist_dislike)

    df_like['if_liked'] = 1
    df_dislike['if_liked'] = 0

    df = pd.concat([df_like, df_dislike], ignore_index=False, sort=True)

    return df
