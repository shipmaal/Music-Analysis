import json


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def song_spotify_info(song_name):
    song = sp.search(song_name, type='track')['tracks']['items'][0]
    uri = song['uri']

    song = sp.track(uri)
    audio_analysis = sp.audio_analysis(uri)
    audio_features = sp.audio_features(uri)

    del song['album']['available_markets']
    del song['available_markets']
    print(json.dumps(song, indent=4))

    return song, audio_analysis, audio_features


def song_length(song_name):
    song = sp.search(song_name, type='track')['tracks']['items'][0]
    return song['duration_ms']
# def find_comp_song(song_name):
#     song = sp.search(song_name, type='track')['tracks']['items'][0]
#     tid = (song['uri'])
#
#     artist_uri = song['album']['artists'][0]['uri']
#     artist_name = song['album']['artists'][0]['name']
#
#     artist_top_tracks = sp.artist_top_tracks(artist_uri, 'US')
#
#     albums = []
#     results = sp.artist_albums(artist_uri, album_type='album')
#     albums.extend(results['items'])
#     while results['next']:
#         results = sp.next(results)
#         albums.extend(results['items'])
#
#     artist_tracks = []
#     for i in albums:
#         artist_tracks.append(sp.album_tracks(i['uri']))
#
#     artist_disco = []
#     for album in artist_tracks:
#         for track in album['items']:
#             if sp.track(track['uri'])['popularity'] > 69:
#                 artist_disco.append(track)
#
#     for i in artist_disco:
#         i.pop('available_markets')
#
#     tracks = {}
#     for i in artist_disco:
#         track = sp.track(i['uri'])
#         if i['uri'] != tid:
#             tracks[i['name']] = {}
#             for j in track:
#                 if j == 'duration_ms' or j == 'popularity' or j == 'uri':
#                     tracks[i['name']][j] = track[j]
#             for k in sp.audio_features(track['id'])[0]:
#                 if k != 'type' and k != 'track_href' and k != 'analysis_url':
#                     tracks[i['name']][k] = sp.audio_features(track['id'])[0][k]
#
#     track_comp = {}
#     for i in tracks:
#         track_comp[i] = {}
#         for j in tracks[i]:
#             if type(tracks[i][j]) == float and 0 <= tracks[i][j] <= 1:
#                 track_comp[i][j] = abs(sp.audio_features(tid)[0][j] - tracks[i][j])
#
#     for i in track_comp:
#         avg = 0.0
#         for j in track_comp[i]:
#             avg += track_comp[i][j]
#         avg = avg / len(track_comp[i])
#         track_comp[i]['average'] = avg
#         track_comp[i]['duration_ms'] = tracks[i]['duration_ms']
#
#     sorted_track_comp = (sorted(track_comp.items(), key=lambda x: x[1]['average']))
#
#     final_tracks = [song]
#     time = song['duration_ms']
#
#     for i in sorted_track_comp:
#         time += i[1]['duration_ms']
#         print(time)
#         if time < 900000:
#             print(i[0])
#             print(i[1]['duration_ms'])
#             final_tracks.append(sp.search(i[0] + ' ' + artist_name, type='track')['tracks']['items'][0])
#             # time += tracks[i[0]]['duration_ms']
#         elif time > 900000:
#             time -= i[1]['duration_ms']
#         if time > 600000:
#             break
#
#     for i in final_tracks:
#         i.pop('available_markets')
#         i['album'].pop('available_markets')
#
#     print(json.dumps(final_tracks, indent=4))
#
#     track_analysis = []
#     for song in final_tracks:
#         track_analysis.append(sp.audio_analysis(song['uri']))
#
#     return final_tracks, track_analysis
