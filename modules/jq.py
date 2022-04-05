"""Everything jobqueue"""

import ujson
import requests
import io
from xml.dom.minidom import parseString
from os import getenv
from collections import namedtuple
from json import JSONEncoder
from dotenv import load_dotenv

load_dotenv()


url = getenv("JOBQUEUE_ENDPOINT")
sub_key = getenv("JOBQUEUE_SUBSCRIPTION_KEY")
check_id = ""
auth = getenv("JOBQUEUE_TOKEN")


def raw(job_id: str = None, endpoint="status"):
    headers = {'Ocp-Apim-Subscription-Key': sub_key}
    raw_url = f"{url}/{endpoint}/{job_id or check_id}"
    response = requests.get(raw_url, headers=headers, data={})
    return response.json()


def status(job_id: str = None):
    return raw(job_id)["values"][0]["status"]


def get_id():
    return raw()["values"][0]["job_id"]


def download(job_id: str = None):
    headers = {'Ocp-Apim-Subscription-Key': sub_key,
               "Authorization": auth}
    download_link = link(job_id)
    r = requests.get(download_link, allow_redirects=True)
    # with open("download.txt", "wb") as f:
    #     f.write(r.content)
    print(r.content)
    return r.content

def valid(content: str) -> bool:
    """Checks if the content is valid. 
    If it's not valid, it means that the file content has XML in it.
    Which is not good. Means something went wrong.

    If it's valid, then the file content is in bytes and can be dumped with JSON.

    Args:
        content (str): file content.

    Returns:
        bool: valid or not.
    """
    try:
        parseString(content)
        return False
    except Exception as e:
        return True


def link(job_id: str = None): 
    return raw(job_id)["values"][0]["result_params"]["download_link"]


def queue(queue="dev", data = None, settings = None):
    """Start a queue.

    Args:
        queue (str, optional): _description_. Defaults to "dev".

    Returns:
        _type_: _description_
    """
    headers = {'Ocp-Apim-Subscription-Key': sub_key}
    raw_url = f"{url}/new/{queue}"
    data = io.BytesIO(ujson.dumps(data).encode())
    response = requests.post(raw_url, params=settings, headers=headers, data=data)
    return response.json()["values"][0]["job_id"]

with open("../pocbot.json", "r") as f:
    output = ujson.load(f)
    new_id = queue(data=output, settings={"compare": "unlabeled"})
    print(new_id)
    print(raw(new_id))
#print(link(new_id))
#content = download(new_id)
#print(valid(content))

