import os
import logging
from retry import retry
from .exceptions import InvalidInputException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .exceptions import NoneResultException
from .utils import escape_special_characters
from .openai import OpenAIModel


class OpenRouterAIModel(OpenAIModel):
    """
    Model that talks to the OpenAI service to interact with models.
    """
    _logger = logging.getLogger(__name__)
    _MODELS = {
        "google/gemini-2.0-flash-lite-preview-02-05:free": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "qwen/qwen-vl-plus:free": "qwen/qwen-vl-plus:free",
        "google/gemini-2.0-pro-exp-02-05:free": "google/gemini-2.0-pro-exp-02-05:free",
        "qwen/qwen2.5-vl-72b-instruct:free": "qwen/qwen2.5-vl-72b-instruct:free",
        "openai/gpt-4o": "openai/gpt-4o",
        "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
        "qwen-turbo": "qwen-turbo",
        "gemini-flash-1.5-8b": "gemini-flash-1.5-8b"
    }

    DEFAULT_MODEL_PARAMS = {
        "temperature": 0.7,
    }
    
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize the OpenAIModel with the given model.
        
        Args:
            model_name (str): The name of the model to use.
        """
        model_id = self._name2model_id(model_name)
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        
        self.model = ChatOpenAI(model=model_id, **self.DEFAULT_MODEL_PARAMS)
        self.parser = StrOutputParser()
