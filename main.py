import uvicorn
import ujson
import intent
from os import getenv
from typing import Optional, Any, Dict, AnyStr, List, Union
from fastapi import FastAPI, Request
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

JSONObject = Dict[str, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

@app.get("/")
async def read_root():
    return "API is running"

@app.post("/scores")
async def intent_inputs(data: JSONStructure, threshold: int = 0.6):
    """Compute similarity matrix & scores of the sentences.

    Args:
        data (JSONStructure): the JSON dataset.
        threshold (int, optional): the threshold to check the score on
        higher than the value = similar. Defaults to 0.6.

    Returns:
        List[Dict[str, str]]: a list of dictionaries: 
        `[{"label1": "id1", "label2": "id2", "score": n}, ...]`
    """
    intents = intent.Intent(file = data)
    return intents.compute_labeled_scores_fast(threshold)

if __name__ == "__main__":
    uvicorn.run(app, host=getenv("HOST"), port=int(getenv("PORT")))
