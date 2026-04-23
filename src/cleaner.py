import re
import logging 
import requests
_white_space=re.sub(r"\s+")
_hypen_space=re.sub(r"-\s+")
_non_ascii=re.sub(r"[^\x20-\x7F]+")
_control=re.sub(r"[\x00-\x1F]+")

headers={"user_agent" : "Moziila/5.0"}
request_timeout=15
max_url_workers=8

def clean(text:str) -> str:
    text=_hypen_space.sub(" ",text)
    text=_non_ascii.sub(" ",text)
    text=_control.sub(" ",text)
    text=_white_space.sub(" ",text)
    return text.strip()

_session=requests.Session()
_session.headers.update(headers)
def url_leader(url:str) -> list[Document]:
    try:
        response=_session.get(url,timeout=request_timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error("Failed to fetch URL '%s': %s", url, e)
