import requests


def verify_ollama_model_present(model):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": "Hi!"
    }

    print('validating model exists...')

    response = requests.post(url, json=data)

    if response.status_code == 404:
        print(f"""Erorr message: {response.json()}""")

    assert response.ok


def pull_ollama_model(model):
    url = "http://localhost:11434/api/pull"
    data = {
        "name": model,
    }
    print(f'pulling ollama model {model}...')
    response = requests.post(url, json=data)
    print(response)
