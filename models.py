
import sys
import json
import requests
import pandas as pd
from langchain_ollama.chat_models import ChatOllama


OLLAMA_URL = "http://localhost:11434/"


def ollama_status():
    """
    Checks the status of the Ollama service by making a GET request to the Ollama server (URL).

    Returns:
        bool: True if the Ollama service is running and responding with a 200 status code, False otherwise.
    """
    try:
        response = requests.get(OLLAMA_URL)
        
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"\nOllama is not running @ {OLLAMA_URL}\n\nError: {e}\n")
        return False


def get_available_models():
    """
    Retrieves the list of available models from the Ollama service.

    Returns:
        list: A list of available model names.
    """
    thelist = requests.get(OLLAMA_URL+"/api/tags")
    jsondata = thelist.json()
    models_available = list()

    for model in jsondata["models"]:
        model_name = model["model"].split(':')[0]
        models_available.append(model_name)

    return models_available


def get_model_info():
    """
    Retrieves information about the available models from the Ollama service.

    Returns:
        pd.DataFrame: A DataFrame containing information about each available model, including the model name, size, and number of parameters.
    """
    thelist = requests.get(OLLAMA_URL+"/api/tags")
    jsondata = thelist.json()
    results = []

    for model in jsondata["models"]:
        model_info = {
            "model": model["model"].split(':')[0],
            "size": f'{round(model["size"] / 1_000_000_000, 1)}GB',
            "parameters": model["details"]["parameter_size"]
        }
        results.append(model_info)

    return pd.DataFrame(results)


def select_model():
    """
    Prompts the user to select a model from the list of available models.
    
    Returns:
        str: The selected model name.
    """
    available_models = get_available_models()

    if "mxbai-embed-large" in available_models:
        available_models.remove("mxbai-embed-large")   

    print("\nAvailable LLMs:\n")
    for i, model in enumerate(available_models, start=1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            user_input = int(input("\nEnter the number of the LLM you want to use: "))
            if 1 <= user_input <= len(available_models):
                return available_models[user_input - 1]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


