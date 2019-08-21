import time
import urllib.parse


def current_milli_time():
    return int(round(time.time() * 1000))


def decode_text(text):
    return urllib.parse.unquote_plus(text)
