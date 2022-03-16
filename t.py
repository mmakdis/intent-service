import numpy as np
import requests
import io

def batch_embed(sentences, batch_size=100):
    result = ()
    sentences = list(sentences) # Current code cannot handle generators
    api_url = 'https://ai-connect.wearetriple.com/tfusem'
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        res = requests.post(
            api_url, json=batch,
            headers={'Ocp-Apim-Subscription-Key': '64d5c45e17c446de9589c9f94cfb0754'} # Subscription key for notebooks and testing
        )
        mem_file = io.BytesIO(res.content)
        mem_file.seek(0)

        # noinspection PyTypeChecker
        chunk = np.load(mem_file, allow_pickle=False)
        result += (chunk,)
    
    return np.vstack(result)


sentences = [
    "My name is Mario.",
    "Mijn naam is Mario."
]

res = requests.post(
    url="https://ai-connect.wearetriple.com/tfusem&resp=numpy",
    params={
        'subscription-key': "",
    },
    json=sentences
)
mem_file = io.BytesIO(res.content)
mem_file.seek(0)
result = np.load(mem_file, allow_pickle=False)
print(result)

#print(batch_embed(["hello i'd like to make an order", "hello i'd like to cancel my order"]))