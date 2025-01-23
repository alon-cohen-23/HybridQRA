import os
from llama_index.llms.cohere import Cohere as LlamaIndexCohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

class LLMServiceManager:
    def __init__(self, provider: str, llm_model: str, embedding_model: str):
        self.provider = provider.lower()
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm = None
        self.embed_model = None
        self.settings = None
        self.input_type = "search_query"  # Default to 'search_query'
        self._initialize_services()

    def _initialize_services(self):
        if self.provider == 'cohere':
            self._initialize_cohere_services()
        elif self.provider == 'azure_openai':
            self._initialize_azure_openai_services()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def _initialize_cohere_services(self):
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY environment variable not set.")

        self.llm = LlamaIndexCohere(api_key=api_key, model=self.llm_model)
        self.embed_model = CohereEmbedding(
            api_key=api_key,
            model_name=self.embedding_model,
            input_type=self.input_type  # Use the dynamic input_type here
        )

    def _initialize_azure_openai_services(self):
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        if not all([api_key, endpoint, api_version, self.llm_model]):
            raise EnvironmentError("Azure OpenAI environment variables or deployment name not set.")

        self.llm = LlamaIndexAzureOpenAI(
            azure_deployment=self.llm_model,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.embed_model = AzureOpenAIEmbedding(
            azure_deployment=self.embedding_model,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def get_llm(self):
        return self.llm

    def get_embedding_model(self):
        return self.embed_model


    def set_input_type(self, input_type: str):
        """Allows changing the input_type dynamically"""
        if input_type not in ["search_query", "search_document"]:
            raise ValueError("Invalid input_type. Must be 'search_query' or 'search_document'.")
        self.input_type = input_type
        # Reinitialize the embedding model with the new input_type
        self.embed_model.input_type = input_type
if __name__ == "__main__":
    llama_index_llm = LLMServiceManager("azure_openai", "gpt-4o-sim", "text-embedding-ada-002")
    print (llama_index_llm.get_llm())
