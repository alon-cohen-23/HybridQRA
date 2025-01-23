from utils.utility_functions import create_index_dict_from_df, read_and_concatenate, convert_search_dict_to_index_dict, update_section_with_kwargs
from src.llm_connections import LLMClient

from FlagEmbedding import FlagReranker
from typing import List, Dict, Optional
import yaml
from qdrant_client import QdrantClient
from tqdm import tqdm

from pathlib import Path
import json

current_file = Path(__file__)
repo_root = current_file.resolve().parent.parent
config_path = repo_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


qdrant_config = config['qdrant']

client = QdrantClient(qdrant_config['client'])
dense_model = qdrant_config['dense_model']
sparse_model = qdrant_config['sparse_model']
chunk_size = qdrant_config['chunk_size']

llm_config = config['llm']


class QdrantCollectionManager:
    _collections_file = 'qdrant_collections.json'
    
    def __init__(self):
        self._client = client
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self.collections_input_files = self._load_collections()
    
    def _load_collections(self) -> Dict[str, List[str]]:
        """Load collections from persistent storage."""
        try:
            with open(self._collections_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_collections(self):
        """Save collections to persistent storage."""
        with open(self._collections_file, 'w') as f:
            json.dump(self.collections_input_files, f)
    
    def create_collection(self, collection_name: str):
        """Create a new Qdrant collection."""
        self._client.set_model(self._dense_model)
        self._client.set_sparse_model(self._sparse_model)
        
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=self._client.get_fastembed_vector_params(),
            sparse_vectors_config=self._client.get_fastembed_sparse_vector_params(), 
            on_disk_payload=True
        )
        self.collections_input_files[collection_name] = []
        self._save_collections()
    
    def add_data_to_collection(
        self, 
        collection_name: str, 
        input_files: List[str], 
        text_field: str, 
        metadata_fields: List[str], 
        chunk_size = chunk_size
    ):
        
        df = read_and_concatenate(input_files)
        index_dict = create_index_dict_from_df(df, text_field, metadata_fields)
        
        self._client.add(
            collection_name=collection_name,
            documents=index_dict['documents'],
            metadata=index_dict['metadata'],
            batch_size=chunk_size
        )  
        files_names = [file_path.split('/')[-1] for file_path in input_files]
        self.collections_input_files[collection_name].extend(files_names)
        self._save_collections()
    
    def delete_collection(self, collection_name: str):
        """Delete a collection and its associated files."""
        del self.collections_input_files[collection_name]
        self._client.delete_collection(collection_name=collection_name)
        self._save_collections()
    
    def get_collections(self) -> List[str]:
        """Retrieve all collection names."""
        return list(self.collections_input_files.keys())
    
    def get_collection_files(self, collection_name: str) -> List[str]:
        """Get input files for a specific collection."""
        return self.collections_input_files[collection_name]
    
            
        
reranker = FlagReranker(qdrant_config['reranker'], qdrant_config['reranker_use_fp16']) 
        
class HybridSearcher ():
    
    
    def search(self, collection_name: str, query: str, search_limit=qdrant_config['search_limit']) -> List[Dict[str, List[str]]]:
        " query the Qdrant collection and return the top answers based on the limit."
        search_result = client.query(
        collection_name=collection_name,
        query_text=query,
        query_filter=None,  
        limit=search_limit,  
    )
        
        
        retrieved_answers = [hit.metadata for hit in search_result]
        # organize retrieved context to only two keys: document and metadata.
        retrieved_answers = [convert_search_dict_to_index_dict(item) for item in retrieved_answers]
        
        return retrieved_answers    
    
        
    def search_with_rerank(self, collection_name: str, query: str, reranker_limit = qdrant_config['reranker_limit']) -> List[str]:
        """
        Parameters
        ----------
        query: the query that has been asked in the serach function.
        
        Returns
        -------
        rellevant_contexts: List
        the top 3 paragraphs sorted in a descending oreder based
        on the score of the bge-reranker-v2-m3 reranking model.
        """
        
        raw_contexts = self.search(collection_name,query)
        
        # Compute scores for each context
        scores = [reranker.compute_score([query, str(item)], normalize=True) for item in raw_contexts]
        
        scored_texts = list(zip(scores, raw_contexts))
        
        #Sort the list of tuples by score in descending order
        sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)
        
        sorted_texts = [text for score, text in sorted_scored_texts]
        rellevant_contexts = sorted_texts[:reranker_limit]
        
        return rellevant_contexts
      
    
    def QA_chain (self, collection_name: str, query: str, **kwargs) -> Dict[str, str]:
        """
        Parameters
        ----------
        collection_name : str
            the name of the rellevant Qdrant collection.
        query : str
            the question you want to ask.
        **kwargs: dict, available keys:
            - prompt: instructions to help the llm to provide a quality answer.
            - model: the llm that will be used to generate the answer.
            - provider: The type provider you use, can be only 'azure_openai' or 'cohere'.
        
        Returns
        -------
        qa_dict : Dict
            contains 3 keys: query - the same query from the input.
            context - the context that helped the llm to answer the query.
            answer - the answer that the llm generated.

        """
        # Access openai configuration from YAML
        updated_config = update_section_with_kwargs(llm_config, **kwargs)
        
        provider = llm_config['provider']
        prompt = updated_config['prompt']
        model = updated_config['model']
        llm_client = LLMClient(provider, model)
        
        contexts = self.search_with_rerank(collection_name, query)
        contexts = str(contexts)
    
        messages = [{"role": "system", "content": prompt},
                   {"role": "user", "content": "Question: " + query},
                   {"role": "user", "content": "Contexts: " +contexts}]
        
        
        response = llm_client.generate_response(messages)
        
        qa_dict = {'question': query, 'context': contexts, 'answer': response}
        
        return qa_dict
        

        
if __name__ =='__main__':
 
    q = QdrantCollectionManager()
    
    text_field = "paragraph_text"
    metadata_fields = ['title', 'content_publish_date']
    input_files = ['../data/espn/sample_espn.csv']
    q.create_collection("espn_stories")
    q.add_data_to_collection("espn_stories", input_files, text_field, metadata_fields)
    
    searcher = HybridSearcher()
    query = "How the celtics won NBA chip?"
    print (searcher.QA_chain("espn_stories", query)['answer'])
    
    
   
  
    
   
   
    
    
    
    
   
