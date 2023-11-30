import os

from spotipy import Spotify, SpotifyClientCredentials
from spotify_dl.spotify import (
    fetch_tracks,
    parse_spotify_url,
    validate_spotify_urls,
    get_item_name,
)

client_id = "3a3ad573b72744ee8edaf197a5529f9d"
client_secret = "f4418e8d79654105b3909dda56f3bb70"

os.environ["SPOTIPY_CLIENT_ID"] = client_id
os.environ["SPOTIPY_CLIENT_SECRET"] = client_secret

url = "https://open.spotify.com/episode/7wI07MKZh1hJmhvhaLz0hz?si=IBEZ2zwyTt2pFu9o4Hba5Q"

sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
)

item_type, item_id = parse_spotify_url(url)

