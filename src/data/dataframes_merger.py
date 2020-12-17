import pandas as pd


def merge_playlist_dataframes(sdf, username, playlist_like, playlist_dislike):
    df_like = sdf.create_dataframe(username, playlist_like)
    df_dislike = sdf.create_dataframe(username, playlist_dislike)

    df_like['if_liked'] = 1
    df_dislike['if_liked'] = 0

    df = pd.concat([df_like, df_dislike], ignore_index=False, sort=True)

    return df
