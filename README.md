# Intent Service

an extension service for Wozzbot.

## .env
Check `.env.example`, you need a `Ocp-Apim-Subscription-Key` (`TFUSEM_KEY`)

## REST API

To run it as a REST API, run `python main.py`

The API currently has one route `/scores`, this computes the similarity scores
of labeled inputs.

### Usage

A simple client:

```python
import requests
with open("path_to_json.json", "r") as f:
    json_data = f.read()
    output = requests.post("http://localhost:8000/scores", data=json_data)
    print(output.json())
```

`/scores` accepts JSON

## Module

You can use it as a module too, the REST API uses `intent.py`

```python
import intent
intents = intent.Intent("path_to_data.json")
```

Or instead of providing the path, you can give a Python dictionary and it'll work too.

## Script

The intent service can be used as a script too.

Run `python intent.py path_to_data.json`
