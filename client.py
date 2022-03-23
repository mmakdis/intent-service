import requests
import ujson
import sys

if __name__ == '__main__':
    with open(sys.argv[1], "r") as f:
        output = requests.post("http://localhost:8000/scores", data=f.read()).json()
        print(output)