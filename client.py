"""
A simple client demonstrating the REST API and how it's used
"""
import requests
import sys

if __name__ == '__main__':
    with open(sys.argv[1], "r") as f:
        json_data = f.read()
        output = requests.post("http://localhost:8000/labeled_matrix", data=json_data)
        print(output.json())
