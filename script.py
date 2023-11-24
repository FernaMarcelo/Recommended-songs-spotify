import json
import os
import numpy as np
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from vars import stored_client_id, stored_client_secret
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import csv


data = pd.read_csv('tracks.csv')


orig_data = data.to_numpy()


data = data.drop(['id', 'name', 'popularity', 'explicit', 'artists',
                 'id_artists', 'release_date', 'duration_ms'], axis=1)


scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

client_id = stored_client_id
client_secret = stored_client_secret
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

original_song_features = {}


def get_song_features():
    global original_song_features
    song_url = input(
        "Por favor ingresa la URL de Spotify de la cancion que desees para obtener recomendaciones: ")

    song_id = song_url.split('/')[-1].split('?')[0]


    track_info = sp.track(song_id)
    audio_features = sp.audio_features(song_id)[0]
    original_song_features = audio_features

    track_features = {
        'danceability': audio_features['danceability'],
        'energy': audio_features['energy'],
        'key': audio_features['key'],
        'loudness': audio_features['loudness'],
        'mode': audio_features['mode'],
        'speechiness': audio_features['speechiness'],
        'acousticness': audio_features['acousticness'],
        'instrumentalness': audio_features['instrumentalness'],
        'liveness': audio_features['liveness'],
        'valence': audio_features['valence'],
        'tempo': audio_features['tempo'],
        'time_signature': audio_features['time_signature']
    }
    return track_features


def recommend_songs(audio_features, n=10, filename='recommended_songs.csv'):

    audio_features_normalized = scaler.transform(
        [list(audio_features.values())])


    distances = []
    for i in range(len(data_normalized)):
        distance = np.linalg.norm(
            audio_features_normalized - data_normalized[i])
        distances.append((i, distance))


    distances.sort(key=lambda x: x[1])
    recommendations = []
    for i in range(n):
        index = distances[i][0]
        recommendations.append(index)


    recommended_songs_data = data.iloc[recommendations]


    songs_list = []
    for song in recommended_songs_data.index:
        original_song = find_original_song(song)
        songs_list.append({
            "id": original_song[0],
            "name": original_song[1],
            "artists": original_song[5],
        })

    recommended_songs_data['spotify_url'] = [
        'https://open.spotify.com/track/' + song['id'] for song in songs_list]

    recommended_songs_data['name'] = [song['name'] for song in songs_list]
    recommended_songs_data['artists'] = [
        ', '.join(song['artists'].split(', ')) for song in songs_list]


    recommended_songs_data = pd.concat(
        [recommended_songs_data, pd.DataFrame(audio_features, index=[0])], axis=1)

    print("ORIGINAL SONG: ")
    print('https://open.spotify.com/track/' +
          str(original_song_features['id']))
    print(json.dumps(original_song_features, indent=4))


    recommended_songs_data = recommended_songs_data[[
        'name', 'artists', 'spotify_url', 'danceability', 'energy',	'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    recommended_songs_data.iloc[:-1, :].to_csv(filename, index=False)

    return recommended_songs_data


def find_original_song(id):
    return orig_data[id-2]



song_features = get_song_features()
recommended_songs = recommend_songs(song_features)

