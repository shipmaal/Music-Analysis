import os
from os.path import basename, join

import subprocess
import json

import wandb

import spotify
import lyricsgenius as lg


def make_folder_dict(path, dictionary):
    if not os.path.isfile(path):
        for d in os.scandir(path):
            d_path = d.path
            if dictionary.__contains__(path):
                if os.path.isfile(d_path):
                    if dictionary[path] == {}:
                        dictionary[path] = []
                    dictionary[path].append(d_path)
                else:
                    dictionary[path].update({d_path: {}})
            make_folder_dict(d_path, dictionary[path])

    return dictionary


def format_file_name(path):
    for i in ['/', '?', '|', '"', '<', '>', '*', ':']:
        if path.__contains__(i):
            path = path.replace(i, '')

    return path


class Project:
    def __init__(self, query, ):
        self.formatted_song_name = query
        self.song_name = None
        self.song = None
        self.artist = None
        self.audio_analysis = None
        self.audio_features = None
        self.path = os.path.dirname(__file__)
        self.dirtree = None

        try:
            self.path = join(self.path, "Projects")
            os.mkdir(self.path)
        except FileExistsError as e:
            print(e)

        # change self.path to include project dir for simpler reading

    # grabs data through spotify api,
    # most important is song spotify url which is uses to download audio files through spotdl
    # will implement spotify's audio analysis later on
    def spotify_grabber(self):
        self.song, self.audio_analysis, self.audio_features = spotify.song_spotify_info(self.formatted_song_name)
        self.song_name = self.song['name'].split(' (feat')[0]
        print(self.song_name)
        self.formatted_song_name = format_file_name(self.song_name)
        self.artist = self.song['album']['artists'][0]['name']
        try:
            self.path = join(self.path, self.formatted_song_name)
            os.mkdir(self.path)
        except Exception as e:
            print(e)

    def lyrics_grabber(self, file_name: str):
        try:
            os.mkdir(join(self.path, 'Lyrics'))
        except Exception as e:
            print(e)

        genius = lg.Genius('qup2IxJwgj7IwTsi1SM5fY-alMaAQvRikNkShSo4kLnrP3qxpMmKLc4rYUis5F0F',
                           skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"],
                           remove_section_headers=True)

        artist = genius.search_artist(self.artist, max_songs=0, sort="title")
        song = genius.search_song(self.song_name, artist.name)
        lyrics = song.lyrics

        file_name = file_name.replace('Audio', 'Lyrics').replace('ogg', 'txt')
        f = open(file_name, 'w', errors='ignore', encoding='windows-1252')
        f.write(lyrics)
        f.close()

    def download_audio(self):
        final_folder_path = join(self.path, 'Audio')
        try:
            os.mkdir(final_folder_path)
        except Exception as e:
            pass

        print(self.formatted_song_name)
        subprocess.run(
            f"spotdl --output-format ogg {self.song['external_urls']['spotify']}",
            cwd=final_folder_path,
            shell=True,
            capture_output=True)

        audio_dir = os.listdir(final_folder_path)
        for i in range(len(audio_dir)):
            audio_dir[i] = join(final_folder_path, audio_dir[i])

        audio_file = (audio_dir[0] if audio_dir[0].__contains__('.ogg')
                      else audio_dir[1] if audio_dir[1].__contains__('.ogg') else None)
        print(audio_file)
        os.rename(audio_file, audio_file.replace(basename(audio_file), (basename(self.path) + '.ogg')))

        self.dirtree = make_folder_dict(self.path, {self.path: {}})

    def download_helper(self):
        self.download_audio()
        print(json.dumps(self.dirtree, indent=4))
        print('e')
        audio_dir = self.dirtree[self.path][join(self.path, 'Audio')]
        for f in audio_dir:
            if not f.__contains__('.spotdl-cache') and f.__contains__('.ogg'):
                self.lyrics_grabber(f)

        # recreate dirtree after adding lyrics
        path = join(self.path, 'Projects', self.formatted_song_name)
        # self.dirtree = make_folder_dict(path, {path: {}})

    def grab_lyric_times(self):


        pass
