import numpy as np
import requests
import io
import os
import itertools
import ujson
import sys
import logging
from timeit import default_timer as timer
from tqdm import tqdm
from typing import Any, Union
from dotenv import load_dotenv
from math import factorial

# load variables from the .env file and put them into the OS environment
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", 
    level=logging.INFO,
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

class Intent():
    """The intent service class
    """

    def __init__(self, file: Union[str, dict] = None):
        """Automatically uses the load_data method to load up a JSON file.

        Args:
            file (str, dict): the JSON data's file path or the JSON data itself.
            If the given object is a dict it's automatically used as the dataset
            otherwise (given a str) the file is opened and read.
        """
        if file:
            if type(file) is str:
                self.json_data = self._load_data(file)
            elif type(file) is dict:
                self.json_data = file

    def _load_data(self, file_path: str) -> dict[str, str]:
        """Load the JSON data from the specified file with ultrajson.

        Args:
            file_path (str): the JSON data's file path.

        Returns:
            dict[str, str]: returns a Python dictionary 
        """
        with open(file_path, "r") as data:
            return ujson.load(data)
        
    def get_label(self, input_id: str):
        """Get the label of the ID.
        Why? Avoids typing long syntax everytime it's needed.

        Args:
            input_id (str): the ID.

        Returns:
            _type_: _description_
        """
        return self.json_data["inputs"][input_id]["classifier"]["label"]
    
    def get_input(self, input_id: int):
        """_summary_

        Args:
            input_id (int): _description_

        Returns:
            _type_: _description_
        """
        return self.json_data["inputs"][input_id]["input"]
        
    def batch_embed(self, sentences: list[str], batch_size: int = 100, api_key = None):
        """Encode a given array using the Universal Sentence Encoder (Multilingual) model

        Args:
            sentences (list[str]): the sentences list to encode.
            batch_size (int, optional): _description_. Defaults to 100.
            api_key (str, optional): tfusem's api key - if nothing is provided it looks uses "KEY" in .env

        Returns:
            list[np.ndarray]: _description_
        """
        logger.info(f"Encoding {len(sentences)} sentences...")
        result = ()
        # Current code cannot handle generators
        sentences = list(sentences)
        api_url = 'https://ai-connect.wearetriple.com/tfusem'
        api_key = api_key or os.getenv("KEY")

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            # Subscription key for notebooks and testing
            res = requests.post(
                api_url, json=batch,
                headers={'Ocp-Apim-Subscription-Key': os.getenv("KEY")}
            )
            mem_file = io.BytesIO(res.content)
            mem_file.seek(0)

            chunk = np.load(mem_file, allow_pickle=False)
            result += (chunk,)
        logger.info("Done encoding.")
        return np.vstack(result)    
    
    def calc_combinations(self, n: int, r: int = 2) -> int:
        """Calculate the possible amount of r (default 2) combinations of n

        Meaning: given an integer 756 (n), how many possible combinations of 2 items
        can be made - if n was an iterable (as in: 756 items)?

        Args:
            n (int): the length of an iterable (set or population)
            r (int, optional): subset of n or sample set, combination length. 
            Defaults to 2.

        Returns:
            int: _description_
        """
        result = factorial(n) / (factorial(r) * factorial((n - r)))
        return int(result)
        
    def get_combinations(self, sentences: list, r: int = 2) -> list:
        """Returns the possible r-length combinations of elements in sentences.

        Args:
            sentences (list): can be a list[np.ndarray] or a list[str] -
            whatever you want to use.

        Returns:
            list[tuple(c)]: a list of tuples, each tuple is a combination of 2.
        """
        logger.info(f"~{self.calc_combinations(len(sentences), r)} possible combinations")
        return itertools.combinations(sentences, r)

    def get_labeled_inputs(self, type: str = "chat", label: str = None):
        """_summary_

        Args:
            type (str, optional): _description_. Defaults to "chat".
            label (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # logger.info("get_labeled_inputs() running...")
        input_data = self.json_data
        all_inputs = {}
        for input_id in input_data["inputs"]:
            input_json = input_data["inputs"][input_id]
            if input_label := input_json["classifier"]["label"]:
                all_inputs[input_label] = all_inputs.get(input_label, [])
                all_inputs[input_label].append((input_id, input_json["input"]))
                #all_inputs[input_label].append(input_json["input"])
        return all_inputs
    
    def get_unlabled_inputs(self):
        input_data = self.json_data
        all_inputs = []
        for input_id in input_data["inputs"]:
            input_json = input_data["inputs"][input_id]
            if input_json["classifier"]["label"] is None:
                all_inputs.append((input_id, input_json["input"]))
        return all_inputs

    def all_sentences_id(self, data: dict, only_ids: bool = False):
        """_summary_

        Args:
            data (dict): _description_
            only_ids (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # logger.info("all_sentences_id() running...")
        all_sentences_id = []
        for value in data.values():
            if only_ids:
                all_sentences_id.extend(sentence[0] for sentence in value)
            else:
                all_sentences_id.extend(iter(value))
        return all_sentences_id

    def all_sentences(self, data: dict):
        """_summary_

        Args:
            data (dict): _description_

        Returns:
            _type_: _description_
        """
        all_sentences = []
        for value in data.values():
            all_sentences.extend(sentence[1] for sentence in value)
        return all_sentences

    def get_labeled_permutations(self, inputs: dict):
        """_summary_

        Args:
            inputs (dict): _description_

        Returns:
            _type_: _description_
        """
        permutations = []
        for label, value_ in inputs.items():
            for input in value_:
                permutations.extend(
                    (input[1], value[0][1])
                    for _label, value in inputs.items()
                    if _label != label
                )
        return permutations
    
    def compute_labeled_scores_fast(self, threshold: float = 0.6) -> dict:
        """Compute the similarity scores using the loaded dataset.


        Args:
            threshold (float, optional): the minimum similarity threshold. 
            Anything higher or equal to this given threshold is added to the 
            output.
            Defaults to 0.6.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each dictionary is a
            computed threshold with what it was compared with. Example:
            `{"label": "id", "label2": "id2", "score": 0.6}`
        """
        data = self.get_labeled_inputs()
        sentences_id = self.all_sentences_id(data, only_ids=True)
        matrix = self.batch_embed(self.all_sentences(data))
        matrix_ids = list(zip(sentences_id, matrix))
        permutations = self.get_combinations(matrix_ids)
        output = []
        logger.info("Calculating similarity matrix (score) for each combination...")
        for _ in permutations:
            id1 = _[0][0]
            id2 = _[1][0]
            label = self.get_label(id1)
            label2 = self.get_label(id2)
            if label == label2:
                continue
            score = np.inner(_[0][1], _[1][1])
            if score < threshold:
                continue
            output.append({label: id1, label2: id2, "score": score.item()})
        return output

    def compute_labeled_scores(self, threshold: float = 0.6) -> dict:
        """DEPRECATED! Use compute_labeled_scores_fast() 

        Compute the similarity scores using the loaded dataset.

        Args:
            threshold (float, optional): the minimum similarity threshold. 
            Anything higher or equal to this given threshold is added to the 
            output.
            Defaults to 0.6.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each dictionary is a
            computed threshold with what it was compared with. Example:
            {"label": "id", "label2": "id2", "score": 0.6}
        """
        data = self.get_labeled_inputs()
        # An index of an encoded sentence from matrix can be used
        # with the sentences_id variable to get the sentence data. 
        # the data consists of a tuple: (id, sentence)
        # the first index of the tuple is well, the ID, that can be used 
        # to know where the sentence is in the dataset.
        sentences_id = self.all_sentences_id(data)
        matrix = self.batch_embed(self.all_sentences(data))
        l_matrix = matrix.tolist()
        permutations = self.get_combinations(matrix)
        output = []
        logger.info("Calculating similarity matrix (score) for each combination...")
        for _ in tqdm(permutations):
            score = np.inner(_[0], _[1])
            string_1 = l_matrix.index(_[0].tolist())
            string_2 = l_matrix.index(_[1].tolist())
            l = sentences_id[string_1]
            l2 = sentences_id[string_2]
            label = self.get_label(l[0])
            label2 = self.get_label(l2[0])
            
            if (score >= threshold) and (label != label2):
                # score.item() is used here because score is a numpy.float32 type
                # FastAPI doesn't know what that is because it's not a Python type 
                # So the REST API returns an error that the object is not iterable
                output.append({label: l[0], label2: l2[0], "score": score.item()})
        return output
                
    def compute_unlabeled_scores(self, threshold: float = 0.6):
        """Compute the simiality scores for unlabeled inputs.

        Args:
            threshold (float, optional): the threshold for the scores. Defaults to 0.6.

        Returns:
            _type_: _description_
        """
        data = self.get_unlabled_inputs()
        sentences = []
        sentence_ids = []
        for sentence in data:
            sentence_ids.append(sentence[0])
            sentences.append(sentence[1])
        #result = self.batch_embed(sentences)
        result = np.loadtxt("helpers/test.out")
        matrix_ids = list(zip(sentence_ids, result))
        #combinations = list(self.get_combinations(matrix_ids))
        output = []
        for s1 in tqdm(range(len(matrix_ids))):
            for s2 in range(s1 + 1, len(matrix_ids)):
                score: np.float32 = np.inner(matrix_ids[s1][1], matrix_ids[s2][1])
                if score < threshold:
                    continue
                output.append((
                    self.get_input(matrix_ids[s1][0]), 
                    self.get_input(matrix_ids[s2][0]),
                    score.item()))
        return output


def similarity(string_1: str, string_2: str, score: float = 0.6) -> dict[str, str]:
    """Computes the similarity matrix. High score indicates greater similarity.

    Args:
        string_1 (str): the first string.
        string_2 (str): the second string.
        score (float, optional): Simiality score to check on. Defaults to 0.6.

    Returns:
        bool: if the computed matrix score is higher than the given score parameter, return True. Otherwise return False.
    """
    matrix = Intent().batch_embed([string_1, string_2])
    computed_score = np.inner(matrix[0], matrix[1]).item()
    return {"score": computed_score, "similar": computed_score >= score}


if __name__ == "__main__":
    intent = Intent(sys.argv[1])
    #output = intent.get_labeled_inputs()
    output = intent.compute_unlabeled_scores()
    print(len(output))
