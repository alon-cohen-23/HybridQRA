import os
from abc import ABC, abstractmethod
from openai import AzureOpenAI
import cohere
from dotenv import load_dotenv


load_dotenv()


# Abstract Strategy Interface
class LLMStrategy(ABC):
    @abstractmethod
    def generate_response(self, messages: list) -> str:
        pass

# Concrete Strategy for Azure OpenAI
class AzureOpenAIStrategy(LLMStrategy):
    def __init__(self, deployment_model: str):
        self.client = AzureOpenAI(
            azure_deployment = deployment_model,
            api_key = os.environ['AZURE_OPENAI_API_KEY'],
            azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
            api_version = os.environ['AZURE_OPENAI_API_VERSION'],
        )

    def generate_response(self, messages: list) -> str:
        url = str(self.client.base_url)
        azure_deployment = url.rstrip('/').split('/')[-1]
        
        response = self.client.chat.completions.create(
            model=azure_deployment,
            messages=messages
        )
        return response.choices[0].message.content

# Concrete Strategy for Cohere
class CohereStrategy(LLMStrategy):
    def __init__(self, model: str = "command-r-plus-08-2024"):
        self.client = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])
        self.model = model

    def generate_response(self, messages: list) -> str:
        response = self.client.chat(
            model=self.model,
            messages = messages
        )
        return response.message.content[0].text.strip()

class LLMClient:
    def __init__(self, provider: str, model):
        """
        Initializes the LLM client based on the provider.

        Args:
            provider (str): The LLM provider ("azure_openai" or "cohere").
            **kwargs: Additional arguments required for the specific provider.
                For Azure OpenAI:
                    - deployment_model (str): The deployment model name.
                For Cohere:
                    - model (str): The model name (default: "command-r-plus-08-2024").
        """
        

        if provider == "azure_openai":
            self.strategy = AzureOpenAIStrategy(model)
        elif provider == "cohere":
            self.strategy = CohereStrategy(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(self, messages: list) -> str:
        """
        Generates a response using the selected LLM strategy.

        Args:
            messages (list): List of messages to send to the LLM.

        Returns:
            str: The generated response.
        """
        return self.strategy.generate_response(messages)


# Usage Example
if __name__ == "__main__":
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
    ]

    client = LLMClient("azure_openai", "gpt-4o-sim")
    print (client.generate_response(messages))
    
    

   
  
