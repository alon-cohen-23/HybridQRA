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

# Context Class
class LLMContext:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: LLMStrategy):
        self.strategy = strategy

    def get_response(self, messages: list) -> str:
        return self.strategy.generate_response(messages)

# Usage Example
if __name__ == "__main__":
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
    ]

    # Using Azure OpenAI
    azure_strategy = AzureOpenAIStrategy(deployment_model="gpt-4o-sim")
    llm_context = LLMContext(strategy=azure_strategy)
    print("Azure OpenAI Response:")
    print(llm_context.get_response(messages))

    

    # Switching to Cohere
    cohere_strategy = CohereStrategy(model="command-r-plus-08-2024")
    llm_context.set_strategy(cohere_strategy)
    print("\nCohere Response:")
    print(llm_context.get_response(messages))
    

   
  
