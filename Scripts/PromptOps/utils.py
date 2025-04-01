import openai
import os

def set_openai_api_key(api_key=None):
    """
    Set the OpenAI API key.
    
    Parameters:
    api_key (str): The OpenAI API key. If not provided, it will be fetched from the environment variable OPENAI_API_KEY.
    """
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OpenAI API key is not set. Please provide an API key.")

def set_llama_api_key(api_key=None):
    """
    Set the Llama API key (Hugging Face token).
    
    Parameters:
    api_key (str): The Hugging Face API token for Llama. If not provided, it will be fetched from the environment variable HUGGINGFACE_API_TOKEN.
    """
    huggingface_api_token = api_key or os.getenv("HUGGINGFACE_API_TOKEN")
    if huggingface_api_token is None:
        raise ValueError("Llama API key is not set. Please provide an API key.")
    return huggingface_api_token

def set_gemini_api_key(api_key=None):
    """
    Set the Gemini (Google Cloud) API key.
    
    Parameters:
    api_key (str): The Google Cloud API key for Gemini. If not provided, it will be fetched from the environment variable GOOGLE_API_KEY.
    """
    google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("Gemini API key is not set. Please provide an API key.")
    return google_api_key

