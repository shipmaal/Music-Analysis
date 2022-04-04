import pandas as pd
import urllib3

import requests
from bs4 import BeautifulSoup as Soup

from pprint import pprint
from json import dumps


client_access_token = 'ZwlHEGWugTC1j-5LaS4tBbTqEWNyQQM4bb1NnSRVZOgUFhReMIPHqo4AUyapUrVc'
base_url = "https://api.genius.com"
token = f"Bearer {client_access_token}"
headers = {'Authorization': token}


def get_songs(artist_name: str):
    querystring = {'q': artist_name}
    response = requests.request("GET", base_url + '/search', headers=headers, params=querystring)
    data = response.json()

    artist_path = (data['response']['hits'][0]['result']['primary_artist']['api_path'])
    artist_url_name = data['response']['hits'][0]['result']['primary_artist']['url'].split('/')[-1]
    print(artist_url_name)
    print(artist_path)

    response = requests.request("GET", base_url + artist_path + "/songs?sort=popularity&per_page=32", headers=headers)
    data = response.json()

    urlist = []
    for i in data['response']['songs']:
        if i['url'].__contains__(artist_url_name):
            urlist.append(i['url'])

    return urlist


http = urllib3.PoolManager()


def get_lyrics(url: str):
    resp = http.urlopen("GET", url)
    print(url)

    soup = Soup(resp.data, 'html.parser')
    print(soup.prettify())
    text = (soup.find_all('span', {"class": "ReferentFragmentVariantdesktop__Highlight-sc-1837hky-1 jShaMP"}))

    lyrics = []
    for i in text:
        t = i.get_text(separator='\n')
        t = t.replace('(', '')
        t = t.replace(')', '')
        print(t)
        if t.__contains__('\n'):
            lyrics.extend(t.split('\n'))
        else:
            lyrics.append(t)

    print(lyrics)

    i = 0
    while i < len(lyrics):
        print(lyrics[i])
        print(lyrics[i])
        if lyrics[i] == '':
            del lyrics[i]
        elif lyrics[i][0] == '[' and lyrics[i][len(lyrics[i]) - 1] == ']':
            del lyrics[i]
            i -= 1
        i += 1

    return lyrics


def panda_shit(lyriclist: list):
    df = pd.DataFrame(lyriclist)
    pass


urlist = get_songs("Tyler the creator")
data = get_lyrics(urlist[0])
