from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
import json
import requests
from utils.logger import get_logger

from assignment_chat.prompts import return_instructions

_logs = get_logger(__name__)


@tool
def get_music_info(entity:str, query:str) -> str:
    """
    Returns music info from the MusicBrainz API.
    """
    _logs.debug(f"*** get_music_info - start: entity={entity}, query={query}")
    api_root_url = "https://musicbrainz.org/ws/2/"
    url = f"{api_root_url}{entity}/"

    params = {
        "query": query,
        "fmt": "json",
        "limit": 1
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)

    _logs.debug(f"*** get_music_info - resp length: {len(resp_dict)}")
    return resp_dict

