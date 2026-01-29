import requests

url = "http://localhost:8000/predict"
data = {
    "texts": ["I am feeling excited today!"]
    }

response = requests.post(url, json = data)
print(response.json())
