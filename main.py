import uvicorn
import ujson
import intent
from os import getenv
from typing import Optional, Any, Dict, AnyStr, List, Union
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from . import worker


load_dotenv()
app = FastAPI()

JSONObject = Dict[str, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

@app.get("/")
async def read_root():
    return "API is running"


@app.post("/labeled_matrix")
async def labeled_inputs(data: JSONStructure, threshold: int = 0.6):
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


@app.post("/unlabeled_matrix")
async def unlabeled_inputs(data: JSONStructure, threshold: int = 0.6):
    """Compute the similarity matrix (scores) of unlabeled inputs.

    Args:
        data (JSONStructure): the JSON dataset.
        threshold (int, optional): the threshold to check the score on.
        higher than the value means it'is similar. Defaults to 0.6.

    Returns:
        List[Dict[str, str]]: a list of dictionaries: 
        `[{"label1": "id1", "label2": "id2", "score": n}, ...]`
    """
    intents = intent.Intent(file = data)
    return intents.compute_unlabeled_scores(threshold)


@app.post("/similarity")
async def unlabeled_inputs(sentence_1: str, sentence_2: str, threshold: float = 0.6):
    """Compares 2 sentences and returns the similarity.

    Args:
        string_1 (str): the first string.
        string_2 (str): the second string.
        score (float, optional): Simiality score to check on. Defaults to 0.6.

    Returns:
        bool: if the computed matrix score is higher than the given score parameter, return True. Otherwise return False.
    """
    return intent.similarity(sentence_1, sentence_2, threshold)
    

if __name__ == "__main__":
    uvicorn.run(app, host=getenv("HOST"), port=int(getenv("PORT")))
