# Intent Service

an extension service for Wozzbot.

## .env
Check `.env.example`, you need a `Ocp-Apim-Subscription-Key` (`TFUSEM_KEY`)

## REST API

To run it as a REST API, run `python main.py`

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
