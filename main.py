import numpy as np
import requests
import io
import itertools
import ujson
import json
import sys

def get_permutations(sentences: list) -> list:
    return list(itertools.permutations(sentences, 2))


def load_data():
    with open(sys.argv[1]) as data:
        return ujson.load(data)
    
def get_inputs(type: str = "chat", label: str = None):
    input_data = load_data()
    all_inputs = []
    for input_id in input_data["inputs"]:
        input_json = input_data["inputs"][input_id]
        if input_json["classifier"]["label"]:
            all_inputs.append(input_json["input"])
    return all_inputs
        

def batch_embed(sentences, batch_size=100):
    result = ()
    # Current code cannot handle generators
    sentences = list(sentences)
    api_url = 'https://ai-connect.wearetriple.com/tfusem'
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # Subscription key for notebooks and testing
        res = requests.post(
            api_url, json=batch,
            headers={'Ocp-Apim-Subscription-Key': '64d5c45e17c446de9589c9f94cfb0754'}
        )
        mem_file = io.BytesIO(res.content)
        mem_file.seek(0)

        chunk = np.load(mem_file, allow_pickle=False)
        result += (chunk,)
    
    return np.vstack(result)

def similarity(string_1: str, string_2: str, score: float = 0.6) -> bool:
    """Computes the similarity matrix. High score indicates greater similarity.

    Args:
        string_1 (str): the first string.
        string_2 (str): the second string.
        score (float, optional): Simiality score to check on. Defaults to 0.6.

    Returns:
        bool: if the computed matrix score is higher than the given score parameter, return True. Otherwise return False.
    """
    matrix = batch_embed([string_1, string_2])
    computed_score = np.inner(matrix[0], matrix[1])
    if computed_score > score:
        print(f"'{string_1}' and '{string_2}' are {computed_score} in similarity")
        return True
    return False

#print(simiality("hello i'd like to make an order", "hallo ik wil m'n bestelling plaatsen"))

if __name__ == "__main__":    
    data = get_inputs()
    result = batch_embed(data).tolist()
    permutations = get_permutations(result)
    for perm in permutations:
        if score := np.inner(perm[0], perm[1]):
            if score > 0.6:
                perm_index_1 = result.index(perm[0])
                perm_index_2 = result.index(perm[1])
                print(f"'{data[perm_index_1]}' and '{data[perm_index_2]}'")
    